"""
LLM 디코딩 결정 시각화 인터페이스
Greedy Search vs Beam Search 기반 단어 선택 과정의 실시간 시각화
"""

import math
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import streamlit as st

from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings('ignore')

# ----------------------------
# Configuration
# ----------------------------
AVAILABLE_MODELS = {
    "distilgpt2": "DistilGPT2 (빠름, 가벼움)",
    "gpt2": "GPT2 (더 나은 품질, 느림)",
    "gpt2-medium": "GPT2-Medium (고품질, 매우 느림)"
}

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 42):
    """재현성을 위한 시드 설정"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_token_str(tokenizer, token_id: int) -> str:
    """토큰을 안전하게 문자열로 변환"""
    try:
        s = tokenizer.decode([token_id])
        # 보기 좋게 공백/개행 정리
        s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        return s if s.strip() else f"<token_{token_id}>"
    except Exception:
        return f"<token_{token_id}>"


def topk_from_logits(logits: torch.Tensor, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    logits에서 상위 k개 토큰 추출
    
    Args:
        logits: [vocab_size] 크기의 텐서
        k: 추출할 토큰 개수
    
    Returns:
        (topk_ids, topk_probs): numpy 배열
    """
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_ids = torch.topk(probs, k=min(k, probs.numel()))
    return topk_ids.detach().cpu().numpy(), topk_probs.detach().cpu().numpy()


@dataclass
class StepTokenInfo:
    """각 생성 단계의 토큰 정보"""
    step: int
    chosen_token_id: int
    chosen_token_str: str
    chosen_prob: float
    top_ids: List[int]
    top_probs: List[float]
    cumulative_logprob: float


@dataclass
class BeamState:
    """Beam Search 상태"""
    token_ids: List[int]
    score: float  # 누적 로그확률 합


@dataclass
class BeamStepInfo:
    """Beam Search 각 단계 정보"""
    step: int
    kept_beams: List[BeamState]
    pruned_count: int
    explored_count: int


# ----------------------------
# Model Loading
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """모델과 토크나이저 로딩 (캐싱됨)"""
    try:
        with st.spinner(f" {model_name} 모델 로딩 중..."):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # EOS 토큰이 없으면 PAD 토큰 사용
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token_id = tokenizer.pad_token_id
            
            model.eval()
            return tokenizer, model
    except Exception as e:
        st.error(f" 모델 로딩 실패: {str(e)}")
        st.stop()


# ----------------------------
# Core Decoding Functions
# ----------------------------
@torch.no_grad()
def next_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """
    다음 토큰의 logits 계산
    
    Args:
        model: 언어 모델
        input_ids: [1, seq_len] 입력 토큰 시퀀스
    
    Returns:
        [vocab_size] 크기의 logits
    """
    out = model(input_ids=input_ids)
    return out.logits[0, -1, :]


@torch.no_grad()
def greedy_decode(
    tokenizer, 
    model, 
    prompt_text: str, 
    max_new_tokens: int, 
    top_n: int,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Greedy Search 디코딩
    
    매 단계마다 가장 높은 확률의 토큰만 선택
    """
    try:
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=True)
        generated = input_ids.clone()
        
        steps: List[StepTokenInfo] = []
        cumulative_logprob = 0.0
        
        for step in range(1, max_new_tokens + 1):
            if progress_callback:
                progress_callback(step, max_new_tokens)
            
            logits = next_logits(model, generated)
            probs = F.softmax(logits, dim=-1)
            
            chosen_id = int(torch.argmax(probs).item())
            chosen_prob = float(probs[chosen_id].item())
            cumulative_logprob += float(torch.log(probs[chosen_id] + 1e-12).item())
            
            top_ids, top_probs = topk_from_logits(logits, top_n)
            
            info = StepTokenInfo(
                step=step,
                chosen_token_id=chosen_id,
                chosen_token_str=safe_token_str(tokenizer, chosen_id),
                chosen_prob=chosen_prob,
                top_ids=top_ids.tolist(),
                top_probs=top_probs.tolist(),
                cumulative_logprob=cumulative_logprob,
            )
            steps.append(info)
            
            # 다음 토큰 추가
            next_id_tensor = torch.tensor([[chosen_id]])
            generated = torch.cat([generated, next_id_tensor], dim=1)
            
            # EOS 토큰이면 중단
            if tokenizer.eos_token_id is not None and chosen_id == tokenizer.eos_token_id:
                break
        
        text_out = tokenizer.decode(generated[0], skip_special_tokens=True)
        
        return {
            "text": text_out,
            "steps": steps,
            "final_logprob": cumulative_logprob,
            "final_prob": float(math.exp(cumulative_logprob)),
        }
    
    except Exception as e:
        st.error(f" Greedy 디코딩 중 오류: {str(e)}")
        return None


@torch.no_grad()
def beam_decode(
    tokenizer, 
    model, 
    prompt_text: str, 
    max_new_tokens: int, 
    beam_width: int, 
    top_n: int,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Beam Search 디코딩
    
    각 단계마다 beam_width개의 후보를 유지하며 탐색
    
    주의: 계산 효율을 위해 각 beam에서 상위 beam_width개 토큰만 확장합니다.
    (표준 Beam Search는 전체 vocab 확장 후 상위 beam_width개 유지)
    """
    try:
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=True)
        base_ids = input_ids[0].tolist()
        
        # 초기 beam: 빈 시퀀스
        beams: List[BeamState] = [BeamState(token_ids=[], score=0.0)]
        beam_steps: List[BeamStepInfo] = []
        bestpath_steps: List[StepTokenInfo] = []
        
        total_pruned = 0
        total_explored = 0
        
        for step in range(1, max_new_tokens + 1):
            if progress_callback:
                progress_callback(step, max_new_tokens)
            
            all_candidates: List[BeamState] = []
            explored = 0
            
            # 각 beam 확장
            for b in beams:
                full_ids = base_ids + b.token_ids
                full_tensor = torch.tensor([full_ids], dtype=torch.long)
                logits = next_logits(model, full_tensor)
                
                probs = F.softmax(logits, dim=-1)
                
                # 계산 효율을 위해 상위 beam_width개만 확장
                # (이는 표준 구현과 약간 다르지만 실용적)
                expand_k = min(beam_width * 2, probs.numel())  # 약간 여유있게
                top_ids, top_probs = topk_from_logits(logits, expand_k)
                explored += len(top_ids)
                
                for tid, tp in zip(top_ids, top_probs):
                    tid = int(tid)
                    tp = float(tp)
                    new_score = b.score + math.log(max(tp, 1e-12))
                    all_candidates.append(
                        BeamState(token_ids=b.token_ids + [tid], score=new_score)
                    )
            
            # 상위 beam_width개만 유지
            all_candidates.sort(key=lambda x: x.score, reverse=True)
            kept = all_candidates[:beam_width]
            pruned = max(0, len(all_candidates) - len(kept))
            
            total_pruned += pruned
            total_explored += explored
            
            beam_steps.append(
                BeamStepInfo(
                    step=step,
                    kept_beams=kept,
                    pruned_count=pruned,
                    explored_count=explored,
                )
            )
            
            beams = kept
            
            # Best beam의 step별 확률 분포 기록 (히트맵용)
            best = beams[0]
            if len(best.token_ids) > 0:
                # 이번 step 선택 직전 상태
                full_ids_best = base_ids + best.token_ids[:-1]
                full_tensor_best = torch.tensor([full_ids_best], dtype=torch.long)
                logits_best = next_logits(model, full_tensor_best)
                
                probs_best = F.softmax(logits_best, dim=-1)
                chosen_id = best.token_ids[-1]
                chosen_prob = float(probs_best[chosen_id].item())
                
                top_ids_vis, top_probs_vis = topk_from_logits(logits_best, top_n)
                bestpath_steps.append(
                    StepTokenInfo(
                        step=step,
                        chosen_token_id=int(chosen_id),
                        chosen_token_str=safe_token_str(tokenizer, int(chosen_id)),
                        chosen_prob=chosen_prob,
                        top_ids=top_ids_vis.tolist(),
                        top_probs=top_probs_vis.tolist(),
                        cumulative_logprob=best.score,
                    )
                )
            
            # EOS 조기 종료
            if (tokenizer.eos_token_id is not None and 
                len(best.token_ids) > 0 and 
                best.token_ids[-1] == tokenizer.eos_token_id):
                break
        
        # 최종 결과
        best = beams[0]
        full_ids = base_ids + best.token_ids
        text_out = tokenizer.decode(full_ids, skip_special_tokens=True)
        
        # 다양성: 최종 beam들의 고유 출력 개수
        final_texts = []
        for b in beams:
            ids = base_ids + b.token_ids
            final_texts.append(tokenizer.decode(ids, skip_special_tokens=True))
        diversity = len(set(final_texts))
        
        return {
            "text": text_out,
            "bestpath_steps": bestpath_steps,
            "beam_steps": beam_steps,
            "final_logprob": best.score,
            "final_prob": float(math.exp(best.score)),
            "diversity": diversity,
            "total_pruned": total_pruned,
            "total_explored": total_explored,
            "final_candidates": final_texts[:5],  # 상위 5개만
        }
    
    except Exception as e:
        st.error(f" Beam 디코딩 중 오류: {str(e)}")
        return None


# ----------------------------
# Visualization Functions
# ----------------------------
def steps_to_table(steps: List[StepTokenInfo]) -> pd.DataFrame:
    """단계별 정보를 테이블로 변환"""
    rows = []
    for s in steps:
        rows.append({
            "단계": s.step,
            "선택된 토큰": s.chosen_token_str,
            "확률": f"{s.chosen_prob:.4f}",
            "누적 로그확률": f"{s.cumulative_logprob:.3f}",
        })
    return pd.DataFrame(rows)


def build_heatmap_data(
    steps: List[StepTokenInfo], 
    tokenizer
) -> Tuple[np.ndarray, List[str], List[str], List[List[str]]]:
    """히트맵 데이터 구성"""
    num_steps = len(steps)
    if num_steps == 0:
        return np.array([]), [], [], []
    
    top_n = len(steps[0].top_ids)
    
    mat = np.zeros((num_steps, top_n), dtype=float)
    y_labels = [f"Step {s.step}" for s in steps]
    x_labels = [f"Top {i+1}" for i in range(top_n)]
    
    token_texts = [
        [safe_token_str(tokenizer, tid) for tid in s.top_ids] 
        for s in steps
    ]
    
    for i, s in enumerate(steps):
        mat[i, :] = np.array(s.top_probs, dtype=float)
    
    return mat, y_labels, x_labels, token_texts


def plot_heatmap(
    mat: np.ndarray, 
    y_labels: List[str], 
    x_labels: List[str], 
    token_texts: List[List[str]], 
    title: str
):
    """토큰 확률 히트맵 그리기"""
    if mat.size == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(y_labels))))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    
    # 셀에 토큰 문자열 표시 (확률 생략 - 칼라맵으로 표현)
    for i in range(min(mat.shape[0], 20)):  # 최대 20개 step만 텍스트 표시
        for j in range(mat.shape[1]):
            tok = token_texts[i][j]
            if len(tok) > 8:
                tok = tok[:8] + "…"
            
            # 배경색에 따라 텍스트 색상 결정
            text_color = "white" if mat[i, j] > 0.5 else "black"
            ax.text(
                j, i, tok, 
                ha="center", va="center", 
                fontsize=7, color=text_color,
                weight="bold"
            )
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("확률", rotation=270, labelpad=15)
    
    fig.tight_layout()
    return fig


def beam_to_simple_tree_table(
    tokenizer, 
    prompt: str, 
    beam_steps: List[BeamStepInfo], 
    max_show: int = 5
) -> pd.DataFrame:
    """
    Beam 트리를 간단한 테이블로 표현
    (Graphviz 대신 사용 가능)
    """
    rows = []
    for bs in beam_steps:
        shown = bs.kept_beams[:max_show]
        for rank, b in enumerate(shown, 1):
            text = tokenizer.decode(
                tokenizer.encode(prompt, add_special_tokens=False) + b.token_ids,
                skip_special_tokens=True
            )
            rows.append({
                "단계": bs.step,
                "순위": rank,
                "점수": f"{b.score:.3f}",
                "생성 텍스트": text[:100]
            })
    
    return pd.DataFrame(rows)


def try_graphviz_tree(
    tokenizer, 
    prompt: str, 
    beam_steps: List[BeamStepInfo], 
    max_show: int = 3
) -> Optional[str]:
    """
    Graphviz DOT 문자열 생성
    Graphviz가 설치되지 않은 경우 None 반환
    """
    try:
        import graphviz
        
        dot_lines = []
        dot_lines.append('digraph BeamTree {')
        dot_lines.append('rankdir=LR;')
        dot_lines.append('node [shape=box, fontsize=9, fontname="Arial"];')
        
        # Root 노드
        prompt_short = prompt[:40].replace('"', '\\"')
        dot_lines.append(f'root [label="PROMPT\\n{prompt_short}..."];')
        
        prev_nodes = ["root"]
        
        for bs in beam_steps:
            curr_nodes = []
            shown = bs.kept_beams[:max_show]
            
            for i, b in enumerate(shown):
                node_name = f's{bs.step}_{i}'
                text = tokenizer.decode(
                    tokenizer.encode(prompt, add_special_tokens=False) + b.token_ids,
                    skip_special_tokens=True
                )
                text_short = text[:50].replace('"', '\\"').replace('\n', '\\n')
                label = f"Step {bs.step}\\nscore={b.score:.2f}\\n{text_short}"
                dot_lines.append(f'{node_name} [label="{label}"];')
                curr_nodes.append(node_name)
            
            # 간단한 연결 (실제 parent 추적은 복잡하므로 생략)
            for pn in prev_nodes:
                for cn in curr_nodes:
                    dot_lines.append(f'{pn} -> {cn};')
            
            prev_nodes = curr_nodes
        
        dot_lines.append('}')
        return '\n'.join(dot_lines)
    
    except ImportError:
        return None


# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(
        page_title="LLM 디코딩 시각화", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title(" LLM 디코딩 결정 시각화")
    st.markdown("**Greedy Search vs Beam Search** - 단어 선택 과정의 실시간 시각화")
    
    # 사이드바 컨트롤
    with st.sidebar:
        st.header(" 설정")
        
        model_key = st.selectbox(
            "모델 선택",
            options=list(AVAILABLE_MODELS.keys()),
            format_func=lambda x: AVAILABLE_MODELS[x],
            index=0
        )
        
        st.markdown("---")
        
        prompt = st.text_area(
            "프롬프트 입력 (영어 권장)",
            value="I love machine learning because",
            height=80,
            help="생성을 시작할 텍스트를 입력하세요. GPT-2는 영어에 최적화되어 있습니다."
        )
        
        if not prompt.strip():
            st.warning(" 프롬프트를 입력해주세요!")
        
        st.markdown("---")
        
        decoding = st.radio(
            "디코딩 알고리즘",
            options=["Greedy", "Beam"],
            index=1,
            help="Greedy: 매 단계 최고 확률 토큰만 선택\nBeam: 여러 후보를 유지하며 탐색"
        )
        
        beam_width = 1
        if decoding == "Beam":
            beam_width = st.slider(
                "Beam Width",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                help="유지할 후보 경로 수"
            )
        
        st.markdown("---")
        
        max_new_tokens = st.slider(
            "생성할 토큰 수",
            min_value=5,
            max_value=40,
            value=20,
            step=5,
            help="프롬프트 이후 생성할 토큰 개수"
        )
        
        top_n = st.slider(
            "히트맵 토큰 개수",
            min_value=5,
            max_value=30,
            value=15,
            step=5,
            help="히트맵에 표시할 상위 토큰 개수"
        )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            run_btn = st.button(" 실행", use_container_width=True, type="primary")
        with col2:
            compare_btn = st.button(" 비교 실험", use_container_width=True)
        
        st.markdown("---")
        st.caption(" Beam Width = 1일 때 Greedy와 동일합니다")
    
    # 모델 로딩
    if prompt.strip():
        tokenizer, model = load_model(model_key)
    else:
        st.info(" 사이드바에서 설정을 조정한 후 실행 버튼을 눌러주세요.")
        return
    
    # 단일 실행
    if run_btn and prompt.strip():
        set_seed(42)
        
        # 진행 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(step, total):
            progress = step / total
            progress_bar.progress(progress)
            status_text.text(f"생성 중... {step}/{total} 단계")
        
        if decoding == "Greedy":
            result = greedy_decode(
                tokenizer, model, prompt, max_new_tokens, top_n,
                progress_callback=update_progress
            )
        else:
            result = beam_decode(
                tokenizer, model, prompt, max_new_tokens, beam_width, top_n,
                progress_callback=update_progress
            )
        
        progress_bar.empty()
        status_text.empty()
        
        if result is None:
            st.error("디코딩 실패")
            return
        
        # 결과 표시
        st.markdown("---")
        st.subheader(" 생성된 텍스트")
        st.info(result["text"])
        
        # Greedy 결과
        if decoding == "Greedy":
            st.markdown("---")
            st.subheader(" 단계별 토큰 선택")
            df = steps_to_table(result["steps"])
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader(" 토큰 확률 히트맵")
            st.caption("각 단계에서 상위 확률을 가진 토큰들의 분포")
            
            mat, ylab, xlab, tok_texts = build_heatmap_data(result["steps"], tokenizer)
            fig = plot_heatmap(
                mat, ylab, xlab, tok_texts,
                title="Greedy Search - 단계별 토큰 확률 분포"
            )
            if fig:
                st.pyplot(fig)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("최종 로그확률", f"{result['final_logprob']:.3f}")
            with col2:
                st.metric("최종 확률", f"{result['final_prob']:.6e}")
        
        # Beam 결과
        else:
            st.markdown("---")
            st.subheader(" 최고 경로 단계별 선택")
            st.caption("Beam Search에서 최종적으로 선택된 경로의 단계별 토큰")
            df = steps_to_table(result["bestpath_steps"])
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader(" 토큰 확률 히트맵 (최고 경로)")
            mat, ylab, xlab, tok_texts = build_heatmap_data(result["bestpath_steps"], tokenizer)
            fig = plot_heatmap(
                mat, ylab, xlab, tok_texts,
                title="Beam Search (최고 경로) - 단계별 토큰 확률 분포"
            )
            if fig:
                st.pyplot(fig)
            
            st.markdown("---")
            st.subheader(" Beam 탐색 요약")
            
            summary_rows = []
            for bs in result["beam_steps"]:
                summary_rows.append({
                    "단계": bs.step,
                    "유지된 Beam": len(bs.kept_beams),
                    "탐색한 후보": bs.explored_count,
                    "제거된 후보": bs.pruned_count
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            
            # Beam 트리 시각화 시도
            st.markdown("---")
            st.subheader(" Beam Search 트리")
            
            dot_str = try_graphviz_tree(tokenizer, prompt, result["beam_steps"], max_show=3)
            
            if dot_str:
                try:
                    st.graphviz_chart(dot_str)
                except Exception as e:
                    st.warning(f" Graphviz 렌더링 실패: {str(e)}")
                    st.info(" 대신 테이블 형태로 표시합니다.")
                    tree_df = beam_to_simple_tree_table(tokenizer, prompt, result["beam_steps"])
                    st.dataframe(tree_df, use_container_width=True, hide_index=True)
            else:
                st.info(" Graphviz가 설치되지 않아 테이블 형태로 표시합니다.")
                tree_df = beam_to_simple_tree_table(tokenizer, prompt, result["beam_steps"])
                st.dataframe(tree_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("최종 로그확률", f"{result['final_logprob']:.3f}")
            with col2:
                st.metric("다양성", result['diversity'], help="최종 후보 중 고유한 출력 개수")
            with col3:
                st.metric("탐색 후보 수", result['total_explored'])
            with col4:
                st.metric("제거 후보 수", result['total_pruned'])
            
            with st.expander(" 최종 유지된 후보들 (상위 5개)"):
                for i, text in enumerate(result["final_candidates"], 1):
                    st.write(f"**{i}.** {text}")
    
    # 비교 실험
    if compare_btn and prompt.strip():
        st.markdown("---")
        st.header(" 자동 비교 실험: Beam Width = 1, 3, 5")
        
        set_seed(42)
        widths = [1, 3, 5]
        results_data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, w in enumerate(widths):
            status_text.text(f"Beam Width = {w} 실행 중... ({idx+1}/3)")
            
            result = beam_decode(tokenizer, model, prompt, max_new_tokens, w, top_n)
            
            if result:
                results_data.append({
                    "Beam Width": w,
                    "최종 로그확률": f"{result['final_logprob']:.3f}",
                    "다양성": result['diversity'],
                    "탐색 후보 수": result['total_explored'],
                    "제거 후보 수": result['total_pruned'],
                    "생성 텍스트": result['text'][:100]
                })
            
            progress_bar.progress((idx + 1) / 3)
        
        progress_bar.empty()
        status_text.empty()
        
        if results_data:
            # 테이블
            df_compare = pd.DataFrame(results_data)
            st.dataframe(df_compare, use_container_width=True, hide_index=True)
            
            # 그래프
            st.markdown("---")
            st.subheader(" 비교 그래프")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig1, ax1 = plt.subplots(figsize=(5, 4))
                ax1.plot(
                    [r["Beam Width"] for r in results_data],
                    [r["탐색 후보 수"] for r in results_data],
                    marker="o", linewidth=2, markersize=8
                )
                ax1.set_title("Beam Width vs 탐색 후보 수")
                ax1.set_xlabel("Beam Width")
                ax1.set_ylabel("탐색 후보 수")
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
            
            with col2:
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                logprobs = [float(r["최종 로그확률"]) for r in results_data]
                ax2.plot(
                    [r["Beam Width"] for r in results_data],
                    logprobs,
                    marker="o", linewidth=2, markersize=8, color="green"
                )
                ax2.set_title("Beam Width vs 최종 로그확률")
                ax2.set_xlabel("Beam Width")
                ax2.set_ylabel("최종 로그확률")
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
            
            with col3:
                fig3, ax3 = plt.subplots(figsize=(5, 4))
                ax3.plot(
                    [r["Beam Width"] for r in results_data],
                    [r["다양성"] for r in results_data],
                    marker="o", linewidth=2, markersize=8, color="orange"
                )
                ax3.set_title("Beam Width vs 다양성")
                ax3.set_xlabel("Beam Width")
                ax3.set_ylabel("다양성 (고유 출력 수)")
                ax3.grid(True, alpha=0.3)
                st.pyplot(fig3)
            
            st.markdown("---")
            st.info("""
             관찰 포인트
            - Beam Width가 증가하면 탐색 후보 수(계산 비용)가 증가합니다.
            - 하지만 최종 로그확률과 다양성은 특정 지점 이후 크게 개선되지 않을 수 있습니다.
            - 이는 Beam Search의 효과가 상황에 따라 제한적임을 보여줍니다.
            """)


if __name__ == "__main__":

    main()

# 🔧 그래프 한글 깨짐 문제 해결 완료!

## ✅ 개선된 내용

### 1. 자동 한글 폰트 감지
- **Windows**: Malgun Gothic, 나눔고딕 자동 감지
- **Mac**: AppleGothic, Apple SD Gothic Neo 자동 감지  
- **Linux**: NanumGothic, Noto Sans CJK KR 자동 감지

### 2. 클라우드 배포 지원
- `packages.txt` 파일 추가로 Streamlit Cloud에서 자동으로 나눔폰트 설치
- Hugging Face Spaces도 동일하게 지원

### 3. 그래프 레이아웃 개선
- **히트맵**: 동적 크기 조정, 빈칸 제거
- **비교 그래프**: 값 레이블 추가, 더 보기 좋은 색상
- **폰트 크기**: 가독성 향상

---

## 🎨 개선 전/후 비교

### Before (개선 전)
```
❌ 그래프 제목: □□□ □□ □□
❌ 축 레이블: □□□
❌ 빈칸이 많음
❌ 작은 글씨
```

### After (개선 후)
```
✅ 그래프 제목: Beam Width vs 탐색 후보 수
✅ 축 레이블: 탐색 후보 수
✅ 레이아웃 최적화 (빈칸 제거)
✅ 적절한 폰트 크기
✅ 값 레이블 표시
```

---

## 📋 새로 추가된 파일

### `packages.txt`
```
fonts-nanum
fonts-nanum-coding
fonts-nanum-extra
```
- Streamlit Cloud / Hugging Face 배포 시 자동으로 한글 폰트 설치
- 로컬 실행에는 영향 없음

---

## 🖥️ 로컬 실행 시 한글 폰트 설정

### Windows
이미 설치되어 있음! (맑은 고딕, 굴림 등)
→ **자동으로 감지됨**

### Mac
이미 설치되어 있음! (AppleGothic 등)
→ **자동으로 감지됨**

### Linux (Ubuntu)
폰트가 없으면 설치:
```bash
sudo apt-get update
sudo apt-get install fonts-nanum fonts-nanum-coding
```

설치 후:
```bash
# 폰트 캐시 새로고침
rm -rf ~/.cache/matplotlib
python -c "import matplotlib.font_manager; matplotlib.font_manager._rebuild()"
```

---

## 🌐 클라우드 배포 시

### Streamlit Cloud
1. `packages.txt` 파일 포함하여 GitHub에 업로드
2. 자동으로 나눔폰트 설치됨
3. 추가 설정 불필요!

파일 구조:
```
your-repo/
├── app.py
├── requirements.txt
├── packages.txt          ← 이 파일!
└── README.md
```

### Hugging Face Spaces
1. `packages.txt` 파일을 함께 업로드
2. Space 설정에서 `packages.txt` 인식됨
3. 자동 설치!

---

## 🔍 폰트 상태 확인

앱 실행 시 상단에 "ℹ️ 시스템 정보" 섹션에서 확인:

```
✅ 한글 폰트: NanumGothic (그래프 한글 표시 가능)
🖥️ 시스템: Linux
```

폰트가 없으면:
```
⚠️ 한글 폰트를 찾을 수 없습니다. 그래프는 영문으로 표시됩니다.
```

---

## 🎯 히트맵 개선 사항

### 1. 동적 크기 조정
- Step 수에 따라 자동으로 높이 조정
- 너무 크거나 작지 않게 최적화

### 2. 스마트 텍스트 표시
- 확률 10% 이상: 토큰 + 확률 표시
- 확률 10% 미만: 토큰만 표시
- 배경색에 따라 텍스트 색상 자동 조정

### 3. 레이블 개선
```python
ax.set_xlabel('토큰 순위', fontweight='bold')
ax.set_ylabel('생성 단계', fontweight='bold')
```

---

## 📊 비교 그래프 개선 사항

### 1. 값 레이블 추가
각 데이터 포인트 위에 실제 값 표시:
```
  50 ←
  ●
  |
  ●
  30
```

### 2. 더 나은 색상
- 탐색 후보 수: 파란색
- 로그확률: 초록색
- 다양성: 주황색

### 3. 큰 마커
- 마커 크기: 8 → 10
- 테두리 추가 (시각적으로 더 명확)

### 4. 굵은 제목/레이블
```python
fontweight='bold'
```

---

## 🆘 여전히 글자가 깨진다면?

### 1. 폰트 캐시 삭제
```bash
# Python
python -c "import matplotlib; print(matplotlib.get_cachedir())"
# 해당 폴더 삭제

# 또는
rm -rf ~/.cache/matplotlib
```

### 2. 앱 재시작
```bash
streamlit run app.py --server.headless true
```

### 3. 강제로 특정 폰트 설정
`app.py` 수정:
```python
# setup_korean_font() 함수에서
plt.rcParams['font.family'] = 'NanumGothic'  # 강제 지정
```

---

## 📸 스크린샷 예시

### 히트맵 (개선 후)
- ✅ 제목, 축 레이블 모두 한글 표시
- ✅ 토큰 이름 + 확률값
- ✅ 깔끔한 레이아웃

### 비교 그래프 (개선 후)
- ✅ 한글 제목/레이블
- ✅ 값 레이블 표시
- ✅ 3개 그래프 나란히 균형있게

---

## 🎉 테스트 확인

로컬에서 실행 후:
1. [ ] 히트맵 제목이 한글로 표시됨
2. [ ] 축 레이블이 한글로 표시됨
3. [ ] 비교 그래프 3개 모두 한글 표시
4. [ ] 빈칸 없이 꽉 찬 레이아웃
5. [ ] 값 레이블이 표시됨

---

## 💡 추가 팁

### 폰트 크기 조정하고 싶다면
`app.py`에서:
```python
plt.rcParams['font.size'] = 10  # 기본 크기
plt.rcParams['axes.titlesize'] = 14  # 제목 크기
plt.rcParams['axes.labelsize'] = 11  # 레이블 크기
```

### 그래프 스타일 바꾸고 싶다면
```python
plt.style.use('seaborn-v0_8-darkgrid')  # 스타일 적용
```

---

## 📝 변경된 파일 목록

1. ✅ `app.py` - 폰트 설정 및 그래프 개선
2. ✅ `packages.txt` - 클라우드 배포용 (새로 추가)
3. ✅ `FONT_FIX.md` - 이 문서!

---

이제 그래프에서 한글이 완벽하게 표시될 거야! 🎨

문제 있으면 바로 말해줘!

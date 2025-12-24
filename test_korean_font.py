"""
한글 폰트 테스트 스크립트
matplotlib에서 한글이 제대로 표시되는지 확인
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

def test_korean_font():
    print("=" * 60)
    print("한글 폰트 테스트")
    print("=" * 60)
    
    # 시스템 정보
    print(f"\n🖥️  시스템: {platform.system()}")
    print(f"📍 Python 버전: {platform.python_version()}")
    
    # 설치된 한글 폰트 찾기
    print("\n🔍 설치된 한글 폰트 목록:")
    korean_fonts = []
    for font in fm.fontManager.ttflist:
        font_name = font.name
        # 한글 폰트 키워드
        korean_keywords = ['Nanum', 'Malgun', 'Gulim', 'Dotum', 'Batang', 
                          'Gothic', 'Apple', 'Noto', 'NanumGothic']
        if any(keyword in font_name for keyword in korean_keywords):
            if font_name not in korean_fonts:
                korean_fonts.append(font_name)
    
    if korean_fonts:
        print("✅ 한글 폰트 발견:")
        for i, font in enumerate(korean_fonts[:10], 1):  # 상위 10개만
            print(f"   {i}. {font}")
        if len(korean_fonts) > 10:
            print(f"   ... 외 {len(korean_fonts) - 10}개")
    else:
        print("❌ 한글 폰트를 찾을 수 없습니다!")
        print("\n해결 방법:")
        if platform.system() == "Linux":
            print("  sudo apt-get install fonts-nanum")
        elif platform.system() == "Darwin":
            print("  한글 폰트가 기본으로 설치되어 있어야 합니다.")
        else:
            print("  Windows는 기본으로 맑은 고딕이 설치되어 있어야 합니다.")
    
    # 테스트 그래프 생성
    print("\n📊 테스트 그래프 생성 중...")
    
    # 폰트 설정
    if korean_fonts:
        plt.rcParams['font.family'] = korean_fonts[0]
        print(f"✅ 사용 폰트: {korean_fonts[0]}")
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        print("⚠️  기본 폰트 사용 (한글 표시 안 됨)")
    
    plt.rcParams['axes.unicode_minus'] = False
    
    # 간단한 그래프
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 그래프 1: 막대 그래프
    categories = ['카테고리 1', '카테고리 2', '카테고리 3']
    values = [10, 25, 15]
    ax1.bar(categories, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('한글 제목 테스트', fontsize=14, fontweight='bold')
    ax1.set_xlabel('카테고리', fontsize=11)
    ax1.set_ylabel('값', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 그래프 2: 선 그래프
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    ax2.plot(x, y, marker='o', linewidth=2, markersize=8)
    ax2.set_title('한글 축 레이블 테스트', fontsize=14, fontweight='bold')
    ax2.set_xlabel('시간 (초)', fontsize=11)
    ax2.set_ylabel('처리량 (개)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 파일로 저장
    output_file = 'korean_font_test.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 테스트 그래프 저장: {output_file}")
    
    # 화면에 표시 (선택)
    try:
        plt.show()
        print("✅ 그래프가 화면에 표시되었습니다.")
    except:
        print("ℹ️  화면 표시 건너뜀 (서버 환경일 수 있음)")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
    print(f"\n{output_file} 파일을 열어서 한글이 제대로 표시되는지 확인하세요.")
    print("- ✅ 한글이 보인다 → 폰트 설정 성공!")
    print("- ❌ □□□로 보인다 → 폰트 설치 필요")
    print("\n")

if __name__ == "__main__":
    test_korean_font()

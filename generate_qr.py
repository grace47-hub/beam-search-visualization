"""
QR 코드 생성 스크립트
보고서/포스터에 삽입할 QR 코드 생성
"""

try:
    import qrcode
    from PIL import Image
except ImportError:
    print("필요한 패키지를 설치해주세요:")
    print("pip install qrcode[pil]")
    exit(1)

# 웹 앱 URL
URL = "https://beam-search-visualization.streamlit.app"

def create_qr_basic():
    """기본 QR 코드 (흑백)"""
    qr = qrcode.QRCode(
        version=1,  # 1~40, 크기 조절
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # 높은 오류 정정
        box_size=10,  # 각 박스 크기
        border=4,  # 테두리 크기
    )
    
    qr.add_data(URL)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    img.save("qr_basic.png")
    print("✅ qr_basic.png 생성 완료!")

def create_qr_highres():
    """고해상도 QR 코드 (인쇄용)"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=20,  # 더 큰 박스
        border=4,
    )
    
    qr.add_data(URL)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # 고해상도로 저장
    img = img.resize((1000, 1000), Image.Resampling.LANCZOS)
    img.save("qr_highres.png", dpi=(300, 300))
    print("✅ qr_highres.png 생성 완료! (고해상도, 인쇄용)")

def create_qr_colored():
    """컬러 QR 코드"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=15,
        border=4,
    )
    
    qr.add_data(URL)
    qr.make(fit=True)
    
    # 파란색 QR 코드
    img = qr.make_image(fill_color="#1F77B4", back_color="white")
    img.save("qr_blue.png")
    print("✅ qr_blue.png 생성 완료! (파란색)")

def create_qr_with_text():
    """텍스트 포함 QR 코드"""
    from PIL import ImageDraw, ImageFont
    
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=12,
        border=2,
    )
    
    qr.add_data(URL)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    
    # 캔버스 확장 (하단에 텍스트 공간)
    width, height = img.size
    new_img = Image.new('RGB', (width, height + 80), 'white')
    new_img.paste(img, (0, 0))
    
    # 텍스트 추가
    draw = ImageDraw.Draw(new_img)
    
    # 폰트 설정 (시스템 기본 폰트)
    try:
        font_title = ImageFont.truetype("arial.ttf", 24)
        font_url = ImageFont.truetype("arial.ttf", 16)
    except:
        font_title = ImageFont.load_default()
        font_url = ImageFont.load_default()
    
    # 제목
    title = "LLM 디코딩 시각화"
    bbox = draw.textbbox((0, 0), title, font=font_title)
    title_width = bbox[2] - bbox[0]
    draw.text(
        ((width - title_width) / 2, height + 10),
        title,
        fill="black",
        font=font_title
    )
    
    # URL (단축)
    url_short = "beam-search-visualization.streamlit.app"
    bbox = draw.textbbox((0, 0), url_short, font=font_url)
    url_width = bbox[2] - bbox[0]
    draw.text(
        ((width - url_width) / 2, height + 45),
        url_short,
        fill="gray",
        font=font_url
    )
    
    new_img.save("qr_with_text.png", dpi=(300, 300))
    print("✅ qr_with_text.png 생성 완료! (텍스트 포함)")

def main():
    print("=" * 60)
    print("QR 코드 생성기")
    print("URL:", URL)
    print("=" * 60)
    print()
    
    print("생성 중...")
    print()
    
    # 모든 QR 코드 생성
    create_qr_basic()
    create_qr_highres()
    create_qr_colored()
    
    try:
        create_qr_with_text()
    except Exception as e:
        print(f"⚠️  텍스트 포함 QR 생성 실패: {e}")
    
    print()
    print("=" * 60)
    print("🎉 생성 완료!")
    print("=" * 60)
    print()
    print("생성된 파일:")
    print("  • qr_basic.png       - 기본 QR (화면용)")
    print("  • qr_highres.png     - 고해상도 QR (인쇄용)")
    print("  • qr_blue.png        - 컬러 QR (발표용)")
    print("  • qr_with_text.png   - 텍스트 포함 QR (포스터용)")
    print()
    print("사용 방법:")
    print("  1. 보고서 → qr_highres.png 사용")
    print("  2. PPT → qr_blue.png 또는 qr_basic.png")
    print("  3. 포스터 → qr_with_text.png")
    print()

if __name__ == "__main__":
    main()

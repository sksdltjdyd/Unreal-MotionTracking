# main.py - 닌자 게임임 실행 파일

"""
닌자 게임임 제스처 인식 시스템
실행 방법:
    python main.py          # 일반 실행
    python main.py test     # 테스트 모드
"""

from gesture_recognizer import NinjaMasterHandTracker
import sys
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def print_banner():
    """시작 배너 출력"""
    print("""
    ╔═══════════════════════════════════════╗
    ║       닌자 게임 - NINJA SHURIKEN        ║
    ║    Hand Gesture Recognition System    ║
    ╚═══════════════════════════════════════╝
    """)
    print("제스처 명령:")
    print("  • 손가락 튕기기 (Flick) - 수리검 발사")
    print("  • 주먹 쥐기 (Fist) - 근접 공격")
    print("  • 손바닥 밀기 (Palm Push) - 충격파")
    print("  • 원 그리기 (Circle) - 특수기")
    print("\n조작법:")
    print("  • 'q' - 종료")
    print("  • 'd' - 디버그 모드 토글")
    print("-" * 45)

def main():
    """메인 함수"""
    print_banner()
    
    # 명령행 인자 확인
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # 테스트 모드
            print("\n테스트 모드로 실행합니다...")
            from gesture_recognizer import test_mode
            test_mode()
            return
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("\n사용법:")
            print("  python main.py          # 일반 실행")
            print("  python main.py test     # OSC 테스트 모드")
            print("  python main.py --help   # 도움말")
            return
    
    # 일반 실행
    try:
        print("\n제스처 인식을 시작합니다...")
        print("웹캠을 준비해주세요...\n")
        
        # 닌자 마스터 트래커 실행
        tracker = NinjaMasterHandTracker(
            osc_ip="127.0.0.1",
            osc_port=7000
        )
        tracker.run()
        
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        logging.error("예외 발생", exc_info=True)
        print("\n문제가 지속되면 다음을 확인하세요:")
        print("1. 웹캠이 연결되어 있는지")
        print("2. 다른 프로그램이 웹캠을 사용 중인지")
        print("3. MediaPipe가 올바르게 설치되어 있는지")

if __name__ == "__main__":
    main()
# main.py - 닌자 마스터 실행 파일

"""
닌자 마스터 제스처 인식 시스템
실행 방법:
    python main.py          # 일반 실행
    python main.py test     # 테스트 모드
    python main.py --stable # 더 안정적인 설정으로 실행
"""

import sys
import logging
from gesture_recognizer import NinjaMasterHandTracker

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """시작 배너 출력"""
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║       닌자 마스터 - NINJA MASTER                  ║
    ║    Hand Gesture Recognition System                ║
    ║         with Position Tracking                    ║
    ╚═══════════════════════════════════════════════════╝
    """)
    print("제스처 명령:")
    print("  • 손가락 튕기기 (Flick) - 수리검 발사 [위치별]")
    print("  • 주먹 쥐기 (Fist) - 공격 막기")
    print("  • 손바닥 밀기 (Palm Push) - 충격파 [위치별]")
    print("\n위치 트래킹:")
    print("  • LEFT   - 화면 왼쪽 영역")
    print("  • CENTER - 화면 중앙 영역")
    print("  • RIGHT  - 화면 오른쪽 영역")
    print("\n조작법:")
    print("  • 'q' - 종료")
    print("  • 'd' - 디버그 모드 토글")
    print("-" * 55)

def get_stabilizer_settings(mode="normal"):
    """모드별 안정화 설정 반환"""
    settings = {
        "easy": {
            "stability_window": 0.5,      # 0.5초 동안 유지
            "confidence_threshold": 0.75,  # 75% 신뢰도
            "cooldown_time": 0.8          # 0.8초 쿨다운
        },
        "normal": {
            "stability_window": 0.3,      # 0.3초 동안 유지
            "confidence_threshold": 0.8,   # 80% 신뢰도
            "cooldown_time": 0.5          # 0.5초 쿨다운
        },
        "stable": {
            "stability_window": 0.4,      # 0.4초 동안 유지
            "confidence_threshold": 0.85,  # 85% 신뢰도
            "cooldown_time": 0.7          # 0.7초 쿨다운
        },
        "expert": {
            "stability_window": 0.2,      # 0.2초 동안 유지
            "confidence_threshold": 0.7,   # 70% 신뢰도
            "cooldown_time": 0.3          # 0.3초 쿨다운
        }
    }
    return settings.get(mode, settings["normal"])

def main():
    """메인 함수"""
    print_banner()
    
    # 명령행 인자 파싱
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # 테스트 모드
            print("\n테스트 모드로 실행합니다...")
            try:
                from gesture_recognizer import test_mode
                test_mode()
            except ImportError:
                print("테스트 모드를 실행할 수 없습니다. test_osc_communication.py 파일이 필요합니다.")
            return
            
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("\n사용법:")
            print("  python main.py              # 일반 실행 (normal 모드)")
            print("  python main.py test         # OSC 테스트 모드")
            print("  python main.py --easy       # 쉬운 모드 (느린 반응)")
            print("  python main.py --stable     # 안정 모드 (오인식 감소)")
            print("  python main.py --expert     # 전문가 모드 (빠른 반응)")
            print("  python main.py --help       # 도움말")
            return
            
        elif sys.argv[1] == "--easy":
            mode = "easy"
            print("\n쉬운 모드로 실행합니다 (느린 반응, 높은 정확도)")
            
        elif sys.argv[1] == "--stable":
            mode = "stable"
            print("\n안정 모드로 실행합니다 (오인식 최소화)")
            
        elif sys.argv[1] == "--expert":
            mode = "expert"
            print("\n전문가 모드로 실행합니다 (빠른 반응)")
            
        else:
            mode = "normal"
            print(f"\n알 수 없는 옵션: {sys.argv[1]}. 일반 모드로 실행합니다.")
    else:
        mode = "normal"
        print("\n일반 모드로 실행합니다.")
    
    # 선택된 모드의 설정 가져오기
    stabilizer_settings = get_stabilizer_settings(mode)
    
    print(f"\n안정화 설정:")
    print(f"  - 제스처 유지 시간: {stabilizer_settings['stability_window']}초")
    print(f"  - 최소 신뢰도: {stabilizer_settings['confidence_threshold']*100:.0f}%")
    print(f"  - 재사용 대기: {stabilizer_settings['cooldown_time']}초")
    
    # 실행
    try:
        print("\n제스처 인식을 시작합니다...")
        print("웹캠을 준비해주세요...\n")
        
        # 닌자 마스터 트래커 실행
        tracker = NinjaMasterHandTracker(
            osc_ip="127.0.0.1",
            osc_port=7000,
            stabilizer_settings_override=stabilizer_settings
        )
        tracker.run()
        
    except IOError as e:
        print(f"\n오류: {e}")
        print("\n해결 방법:")
        print("1. 웹캠이 연결되어 있는지 확인")
        print("2. 다른 프로그램이 웹캠을 사용 중인지 확인")
        print("3. 웹캠 권한 설정 확인")
        
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
        
    except Exception as e:
        print(f"\n치명적 오류 발생: {e}")
        logging.error("예외 발생", exc_info=True)
        print("\n문제가 지속되면:")
        print("1. Python 패키지가 모두 설치되어 있는지 확인")
        print("   pip install -r requirements.txt")
        print("2. gesture_stabilizer.py 파일이 있는지 확인")
        print("3. 로그를 확인하여 상세 오류 내용 파악")

if __name__ == "__main__":
    main()
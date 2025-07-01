# main.py - 닌자 마스터 실행 파일 (멀티 제스처 버전)

"""
닌자 마스터 제스처 인식 시스템 - 멀티 제스처 개선 버전
실행 방법:
    python main.py          # 기본 실행 (안정화 모드)
    python main.py test     # 테스트 모드
    python main.py --fast   # 빠른 모드
    python main.py --slow   # 느린 모드 (더 안정적)
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
    ║      닌자 마스터 - MULTI-GESTURE EDITION            ║
    ║         Enhanced Hand Gesture                     ║
    ║            Recognition System                     ║
    ╚═══════════════════════════════════════════════════╝
    """)
    print("지원 제스처:")
    print("  • FLICK (손가락 튕기기)")
    print("    - 검지와 중지를 붙인 상태")
    print("    - 아래에서 위로 빠르게 튕기기")
    print("    - 양손 모두 인식")
    print("    - 위치별 액션 지원")
    print("\n  • FIST (주먹 쥐기)")
    print("    - 모든 손가락을 굽히기")
    print("    - 양손 모두 인식")
    print("\n  • PINCH (집기)")
    print("    - 엄지와 검지 끝을 접촉")
    print("    - 양손 모두 인식")
    print("    - 위치별 액션 지원")
    print("\n위치 트래킹:")
    print("  • LEFT   - 화면 왼쪽 영역")
    print("  • CENTER - 화면 중앙 영역")
    print("  • RIGHT  - 화면 오른쪽 영역")
    print("\n조작법:")
    print("  • 'q' - 종료")
    print("  • 'd' - 디버그 모드 토글")
    print("-" * 55)

def get_stabilizer_settings(mode="stable"):
    """모드별 안정화 설정 반환 - 멀티 제스처용"""
    settings = {
        "slow": {
            "stability_window": 0.5,        # 500ms - 매우 안정적
            "confidence_threshold": 0.25,   # 85% 신뢰도 - 매우 엄격
            "cooldown_time": 0.1           # 1000ms 쿨다운
        },
        "stable": {
            "stability_window": 0.3,        # 300ms - 안정적 (기본)
            "confidence_threshold": 0.25,   # 75% 신뢰도
            "cooldown_time": 0.1           # 600ms 쿨다운
        },
        "fast": {
            "stability_window": 0.15,       # 150ms - 빠름
            "confidence_threshold": 0.25,   # 65% 신뢰도
            "cooldown_time": 0.1           # 300ms 쿨다운
        }
    }
    return settings.get(mode, settings["stable"])

def print_gesture_tips():
    """제스처별 팁 출력"""
    print("\n💡 제스처 팁:")
    print("\n[FLICK - 표창 던지기]")
    print("  • 검지와 중지를 확실히 붙이세요")
    print("  • 아래에서 위로 빠르게 튕기세요")
    print("  • 수직 움직임이 60% 이상이어야 합니다")
    
    print("\n[FIST - 주먹 방어]")
    print("  • 엄지를 제외한 4개 손가락을 굽히세요")
    print("  • 3개 이상 손가락이 굽혀져야 인식됩니다")
    
    print("\n[PINCH - 특수 공격]")
    print("  • 엄지와 검지 끝을 가까이 하세요")
    print("  • 나머지 손가락은 펴는 것이 좋습니다")
    print("  • 거리가 0.045 이하여야 합니다")

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
            print("  python main.py          # 기본 실행 (stable 모드)")
            print("  python main.py test     # OSC 테스트 모드")
            print("  python main.py --slow   # 느린 모드 (더 안정적)")
            print("  python main.py --fast   # 빠른 모드 (덜 안정적)")
            print("  python main.py --help   # 도움말")
            return
            
        elif sys.argv[1] == "--slow":
            mode = "slow"
            print("\n느린 모드로 실행합니다 (매우 안정적)")
            
        elif sys.argv[1] == "--fast":
            mode = "fast"
            print("\n빠른 모드로 실행합니다 (덜 안정적)")
            
        else:
            mode = "stable"
            print(f"\n알 수 없는 옵션: {sys.argv[1]}. 안정화 모드로 실행합니다.")
    else:
        mode = "stable"
        print("\n안정화 모드로 실행합니다.")
    
    # 선택된 모드의 설정 가져오기
    stabilizer_settings = get_stabilizer_settings(mode)
    
    print(f"\n안정화 설정:")
    print(f"  - 제스처 유지 시간: {stabilizer_settings['stability_window']*1000:.0f}ms")
    print(f"  - 최소 신뢰도: {stabilizer_settings['confidence_threshold']*100:.0f}%")
    print(f"  - 재사용 대기: {stabilizer_settings['cooldown_time']*1000:.0f}ms")
    
    # 제스처 팁 출력
    print_gesture_tips()
    
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
        print("   pip install opencv-python mediapipe numpy python-osc")
        print("2. gesture_recognizer.py 파일이 있는지 확인")
        print("3. 로그를 확인하여 상세 오류 내용 파악")

if __name__ == "__main__":
    main()
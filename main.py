# main.py - 닌자 마스터 실행 파일

"""
닌자 마스터 제스처 인식 시스템
실행 방법:
    python main.py         # 일반 실행
    python main.py test    # 테스트 모드
    python main.py --stable # 더 안정적인 설정으로 실행
"""

import sys
import logging
# gesture_recognizer.py에서 NinjaMasterHandTracker와 test_mode를 가져옵니다.
from gesture_recognizer import NinjaMasterHandTracker, test_mode as run_gesture_test_mode

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """시작 배너 출력 (3-제스처 버전 기준)"""
    print("""
    ╔═══════════════════════════════════════╗
    ║        닌자 마스터 - NINJA MASTER        ║
    ║     Hand Gesture Recognition System     ║
    ╚═══════════════════════════════════════╝
    """)
    print("제스처 명령 (3-제스처 시스템):")
    print("  • 손가락 튕기기 (Flick) - 수리검 발사")
    print("  • 주먹 쥐기 (Fist) - 공격 막기") # 이전 gesture_recognizer.py에서는 '공격 막기'로 되어 있었음
    print("  • 손바닥 밀기 (Palm Push) - 충격파")
    # print("  • 원 그리기 (Circle) - 특수기") # 4번째 제스처는 현재 gesture_recognizer.py에 없음
    print("\n조작법:")
    print("  • 'q' - 종료")
    print("  • 'd' - 디버그 모드 토글")
    print("-" * 45)

def get_stabilizer_settings(mode="normal"):
    """모드별 안정화 설정 반환"""
    settings = {
        "easy": {
            "stability_window": 0.5,     # 0.5초 동안 유지
            "confidence_threshold": 0.75,  # 75% 신뢰도
            "cooldown_time": 0.8         # 0.8초 쿨다운
        },
        "normal": {
            "stability_window": 0.3,     # 0.3초 동안 유지
            "confidence_threshold": 0.8,   # 80% 신뢰도
            "cooldown_time": 0.5         # 0.5초 쿨다운
        },
        "stable": {
            "stability_window": 0.4,     # 0.4초 동안 유지
            "confidence_threshold": 0.85,  # 85% 신뢰도
            "cooldown_time": 0.7         # 0.7초 쿨다운
        },
        "expert": {
            "stability_window": 0.2,     # 0.2초 동안 유지
            "confidence_threshold": 0.7,   # 70% 신뢰도
            "cooldown_time": 0.3         # 0.3초 쿨다운
        }
    }
    return settings.get(mode, settings["normal"])

def main():
    """메인 함수"""
    print_banner()
    
    mode = "normal" # 기본 모드

    # 명령행 인자 파싱
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower() # 소문자로 비교 일관성 유지
        if arg == "test":
            print("\n테스트 모드로 실행합니다...")
            try:
                # gesture_recognizer.py의 test_mode 함수를 직접 호출
                run_gesture_test_mode()
            except ImportError:
                # gesture_recognizer.py에 test_mode 함수가 없는 경우 또는 파일 자체를 못 찾는 경우
                print("테스트 모드를 실행할 수 없습니다. gesture_recognizer.py 파일 및 test_mode 함수를 확인하세요.")
            except AttributeError:
                 print("테스트 모드를 실행할 수 없습니다. gesture_recognizer.py에 test_mode 함수가 정의되어 있는지 확인하세요.")
            return # 테스트 모드 실행 후 종료
            
        elif arg == "--help" or arg == "-h":
            print("\n사용법:")
            print("  python main.py             # 일반 실행 (normal 모드)")
            print("  python main.py test        # OSC 테스트 모드")
            print("  python main.py --easy      # 쉬운 모드 (느린 반응)")
            print("  python main.py --stable    # 안정 모드 (오인식 감소)")
            print("  python main.py --expert    # 전문가 모드 (빠른 반응)")
            print("  python main.py --help      # 도움말")
            return
            
        elif arg == "--easy":
            mode = "easy"
            print("\n쉬운 모드로 실행합니다 (느린 반응, 높은 정확도)")
            
        elif arg == "--stable":
            mode = "stable"
            print("\n안정 모드로 실행합니다 (오인식 최소화)")
            
        elif arg == "--expert":
            mode = "expert"
            print("\n전문가 모드로 실행합니다 (빠른 반응)")
            
        else:
            print(f"\n알 수 없는 옵션: {sys.argv[1]}. 일반 모드로 실행합니다.")
    else:
        print("\n일반 모드로 실행합니다.")
    
    # 선택된 모드의 설정 가져오기
    stabilizer_settings = get_stabilizer_settings(mode)
    
    print(f"\n안정화 설정 ({mode} 모드):")
    print(f"  - 제스처 유지 시간: {stabilizer_settings['stability_window']}초")
    print(f"  - 최소 신뢰도: {stabilizer_settings['confidence_threshold']*100:.0f}%")
    print(f"  - 재사용 대기: {stabilizer_settings['cooldown_time']}초")
    
    # 실행
    tracker_instance = None
    try:
        print("\n제스처 인식을 시작합니다...")
        print("웹캠을 준비해주세요...\n")
        
        # 닌자 마스터 트래커 실행
        # NinjaMasterHandTracker의 __init__ 파라미터에 맞게 'stabilizer_settings' 사용
        tracker_instance = NinjaMasterHandTracker(
            osc_ip="127.0.0.1", # 필요한 경우 Config에서 가져오거나 직접 설정
            osc_port=7000,      # 필요한 경우 Config에서 가져오거나 직접 설정
            stabilizer_settings=stabilizer_settings # 수정된 파라미터명
        )
        tracker_instance.run()
        
    except IOError as e: # 주로 카메라 관련 오류
        print(f"\n오류: {e}")
        print("\n해결 방법:")
        print("1. 웹캠이 올바르게 연결되어 있고, 다른 프로그램에서 사용하고 있지 않은지 확인해주세요.")
        print("2. 카메라 접근 권한이 애플리케이션에 부여되었는지 확인해주세요.")
        
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
        
    except Exception as e:
        print(f"\n치명적 오류 발생: {e}")
        logging.error("예외 발생", exc_info=True) # 로그에 상세 트레이스백 기록
        print("\n문제가 지속되면 다음을 확인해주세요:")
        print("1. 필요한 Python 패키지 (opencv-python, mediapipe, python-osc, numpy)가 모두 최신 버전으로 설치되어 있는지 확인해주세요.")
        print("   (예: pip install --upgrade opencv-python mediapipe python-osc numpy)")
        print("2. 'gesture_recognizer.py' 파일이 'main.py'와 동일한 디렉토리에 있는지 확인해주세요.") # 파일 이름 수정
        print("3. 콘솔 또는 로그 파일의 상세 오류 내용을 확인하여 문제의 원인을 파악해주세요.")
    finally:
        if tracker_instance and hasattr(tracker_instance, 'cleanup') and callable(tracker_instance.cleanup):
            logger.info("애플리케이션 종료 전 리소스 정리 시도...")
            # tracker_instance.cleanup() # run() 메소드 내부의 finally에서 이미 호출됨
                                     # 여기서 또 호출하면 중복이므로, run() 내부의 cleanup을 신뢰
        logger.info("프로그램 실행 종료.")


if __name__ == "__main__":
    main()
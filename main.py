# main.py - 닌자 마스터 실행 파일 (멀티 제스처 버전)

"""
닌자 마스터 제스처 인식 시스템 - 멀티 제스처 버전 v2.0
실행 방법:
    python main.py          # 빠른 모드 (기본값)
    python main.py --normal # 일반 모드
    python main.py --ultra  # 초고속 모드 (매우 민감)
    python main.py --help   # 도움말
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
    ║      닌자 마스터 - Multi-Gesture Edition v2.0       ║
    ║          Ultra Fast Hand Gesture System           ║
    ╚═══════════════════════════════════════════════════╝
    """)
    print("제스처 명령 (양손 지원):")
    print("  • 손가락 튕기기 (Flick) - 수리검 발사")
    print("    - 검지/중지 붙이고 빠르게 움직이거나 위로 튕기기")
    print("  • 주먹 쥐기 (Fist) - 기 모으기 등")
    print("  • 핀치 (Pinch) - 아이템 선택 등")
    print("    - 엄지와 검지 끝 맞대기")
    print("\n위치 트래킹:")
    print("  • LEFT / CENTER / RIGHT")
    print("\n조작법:")
    print("  • 'q' - 종료")
    print("  • 'd' - 디버그 모드 토글")
    print("-" * 55)

def get_stabilizer_settings(mode="fast"):
    """모드별 안정화 설정 반환"""
    settings = {
        "normal": {
            "stability_window": 0.1,    # 100ms
            "confidence_threshold": 0.7,  # 70% 신뢰도
            "cooldown_time": 0.4        # 400ms 쿨다운
        },
        "fast": {
            "stability_window": 0.05,   # 50ms
            "confidence_threshold": 0.65, # 65% 신뢰도
            "cooldown_time": 0.3        # 300ms 쿨다운
        },
        "ultra": {
            "stability_window": 0.02,   # 20ms - 초고속
            "confidence_threshold": 0.6,  # 60% 신뢰도
            "cooldown_time": 0.2        # 200ms 쿨다운
        }
    }
    return settings.get(mode, settings["fast"])

def main():
    """메인 함수"""
    print_banner()
    
    # 명령행 인자 파싱
    mode = "fast" # 기본 모드
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--help" or arg == "-h":
            print("\n사용법:")
            print("  python main.py          # 빠른 실행 (fast 모드)")
            print("  python main.py --normal # 일반 속도")
            print("  python main.py --ultra  # 초고속 모드")
            print("  python main.py --help   # 도움말")
            return
            
        elif arg == "--normal":
            mode = "normal"
            print("\n일반 모드로 실행합니다.")
            
        elif arg == "--ultra":
            mode = "ultra"
            print("\n초고속 모드로 실행합니다 (매우 민감함).")
            
        else:
            print(f"\n알 수 없는 옵션: {arg}. 기본 'fast' 모드로 실행합니다.")
    else:
        print("\n빠른 모드로 실행합니다.")
    
    # 선택된 모드의 설정 가져오기
    stabilizer_settings = get_stabilizer_settings(mode)
    
    print(f"\n안정화 설정 ({mode} mode):")
    print(f"  - 제스처 유지 시간: {stabilizer_settings['stability_window']*1000:.0f}ms")
    print(f"  - 최소 신뢰도: {stabilizer_settings['confidence_threshold']*100:.0f}%")
    print(f"  - 재사용 대기: {stabilizer_settings['cooldown_time']*1000:.0f}ms")
    
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
        print("\n[해결 방법]")
        print("1. 웹캠이 컴퓨터에 연결되어 있는지 확인하세요.")
        print("2. 다른 프로그램(Zoom, Skype 등)이 웹캠을 사용 중인지 확인하고 종료해주세요.")
        print("3. 운영체제 설정에서 프로그램의 웹캠 접근 권한을 확인해주세요.")
        
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
        
    except Exception as e:
        print(f"\n치명적 오류 발생: {e}")
        logging.error("예외 발생", exc_info=True)
        print("\n[문제 해결 가이드]")
        print("1. 필요한 Python 패키지가 모두 설치되어 있는지 확인하세요.")
        print("   pip install opencv-python mediapipe numpy python-osc")
        print("2. 'gesture_recognizer.py' 파일이 'main.py'와 같은 폴더에 있는지 확인하세요.")
        print("3. 상세한 오류 내용은 로그를 참고하세요.")

if __name__ == "__main__":
    main()
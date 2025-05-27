# Unreal Motion Tracking Game Dev

Open CV - Receive webcam
MediaPipe - Motion tracking
Unreal - Based on Blue print and OSC Plug-in

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

닌자 게임(수리검 던지기) - 제스처 인식 시스템
🚀 빠른 시작
1. 환경 설정
bash# 가상환경 생성
python -m venv ninja_env

# 가상환경 활성화
# Windows:
ninja_env\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
2. 실행
bash# 일반 실행
python main.py

# OSC 테스트 모드
python main.py test


🎮 제스처 명령
제스처

손가랑 튕기기 --> 검지를 빠르게 튕기기 --> 수리검 발사

주먹 쥐기 --> 주먹 쥐기 --> 근접공격

손바닥 밀기 --> 손바닥을 앞으로 밀기 --> 충격파

원 그리기 --> 검지로 원그리기 --> 특수기


🔧 OSC 통신
메시지 형식

포트: 7000
주소 체계: /ninja/*

주요 메시지
/ninja/gesture/type [string]      # 제스처 타입
/ninja/gesture/confidence [float] # 신뢰도 (0-1)
/ninja/gesture/direction [x, y]   # 방향 벡터
/ninja/gesture/speed [float]      # 속도 (pixels/sec)
/ninja/hand/detected [int]        # 손 감지 여부
/ninja/hand/count [int]          # 감지된 손 개수


🧪 테스트
OSC 통신 테스트
bashpython test_osc_communication.py

테스트 옵션:
1. 모든 제스처 테스트
2. 손 추적 테스트
3. 게임플레이 시뮬레이션
4. 랜덤 제스처 생성
5. 수동 테스트 모드


📊 성능 최적화
목표 FPS: 30 FPS
입력 지연: < 50ms
제스처 인식률: > 95%


🐛 문제 해결
웹캠이 열리지 않을 때
1. 다른 프로그램이 웹캠을 사용 중인지 확인
2. 웹캠 권한 설정 확인
3. 다른 카메라 인덱스 시도

OSC 메시지가 전송되지 않을 때
1. 방화벽 설정 확인
2. 포트 7000이 사용 중인지 확인
3. 언리얼 엔진의 OSC 플러그인 활성화 확인

제스처가 인식되지 않을 때
1. 조명이 충분한지 확인
2. 카메라와의 거리 조정 (0.5-2m)
3. 손이 화면에 완전히 보이는지 확인


📝 개발 노트
Python 3.9+ 권장
CUDA 지원 GPU가 있으면 성능 향상
디버그 모드: 실행 중 'd' 키 토글


📄 라이선스
이 프로젝트는 닌자 수리검 던지기를 위해 개발되었습니다다
# gesture_stabilizer.py - 제스처 안정화 모듈

import time
from collections import deque
from enum import Enum

class GestureType(Enum):
    """제스처 타입 정의"""
    NONE = "none"
    IDLE = "idle"
    FLICK = "flick"
    FIST = "fist"
    PALM_PUSH = "palm_push"
    CROSS_BLOCK = "cross_block"
    CIRCLE = "circle"

class GestureStabilizer:
    """제스처 안정화 및 노이즈 필터링"""
    
    def __init__(self, 
                 stability_window=0.3,      # 제스처 유지 시간 (초)
                 confidence_threshold=0.8,   # 최소 신뢰도
                 cooldown_time=0.5,         # 제스처 간 쿨다운
                 gesture_buffer_size=5):    # 버퍼 크기
        
        self.stability_window = stability_window
        self.confidence_threshold = confidence_threshold
        self.cooldown_time = cooldown_time
        
        # 제스처 히스토리
        self.gesture_buffer = deque(maxlen=gesture_buffer_size)
        self.last_sent_gesture = "none"
        self.last_sent_time = 0
        
        # 제스처별 쿨다운
        self.gesture_cooldowns = {}
        
        # 안정화 타이머
        self.current_gesture_start = 0
        self.current_gesture_type = "none"
        
        print(f"GestureStabilizer 초기화: window={stability_window}s, threshold={confidence_threshold}, cooldown={cooldown_time}s")
    
    def should_send_gesture(self, gesture_type, confidence, hand_label="Right"):
        """제스처를 전송해야 하는지 판단"""
        
        current_time = time.time()
        
        # 1. None 제스처는 전송하지 않음
        if gesture_type == "none":
            self.current_gesture_type = "none"
            return False, None
        
        # 2. 신뢰도 체크
        if confidence < self.confidence_threshold:
            return False, None
        
        # 3. 쿨다운 체크
        gesture_key = f"{gesture_type}_{hand_label}"
        if gesture_key in self.gesture_cooldowns:
            if current_time - self.gesture_cooldowns[gesture_key] < self.cooldown_time:
                return False, None
        
        # 4. 안정성 체크 - 제스처가 일정 시간 유지되어야 함
        if gesture_type != self.current_gesture_type:
            # 새로운 제스처 시작
            self.current_gesture_type = gesture_type
            self.current_gesture_start = current_time
            return False, None  # 아직 전송하지 않음
        
        # 5. 충분한 시간 유지됐는지 확인
        time_held = current_time - self.current_gesture_start
        if time_held < self.stability_window:
            return False, None  # 아직 대기 중
        
        # 6. 이미 전송한 제스처인지 확인
        if gesture_type == self.last_sent_gesture:
            # 같은 제스처는 쿨다운 시간 후에만 재전송
            if current_time - self.last_sent_time < self.cooldown_time:
                return False, None
        
        # 7. 제스처 전송 승인
        self.last_sent_gesture = gesture_type
        self.last_sent_time = current_time
        self.gesture_cooldowns[gesture_key] = current_time
        
        return True, {
            "type": gesture_type,
            "confidence": confidence,
            "hand": hand_label,
            "timestamp": current_time
        }
    
    def reset_if_idle(self):
        """손이 없거나 IDLE 상태일 때 리셋"""
        self.current_gesture_type = "none"
        self.current_gesture_start = 0
    
    def get_statistics(self):
        """디버그용 통계"""
        current_time = time.time()
        
        # 현재 제스처가 유지된 시간
        if self.current_gesture_type != "none" and self.current_gesture_start > 0:
            stability_progress = current_time - self.current_gesture_start
        else:
            stability_progress = 0
        
        return {
            "last_gesture": self.last_sent_gesture,
            "time_since_last": current_time - self.last_sent_time if self.last_sent_time > 0 else 0,
            "current_gesture": self.current_gesture_type,
            "stability_progress": stability_progress
        }
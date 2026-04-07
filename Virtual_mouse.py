"""
Hand Gesture Mouse Controller v4
==================================
MediaPipe Tasks API (0.10.30+)

Gesture map:
  Index up only                     → move mouse (index tip)
  Thumb + Middle pinch tap  (<0.35s)→ left click
  Thumb + Middle pinch hold (>0.55s)→ right click
  Index + Middle up, hand moves down→ scroll down (continuous)
  Index + Middle up, hand moves up  → scroll up   (continuous)
  Anything else                     → freeze mouse
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pyautogui
import numpy as np
import time
import urllib.request
import os

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SMOOTHING_ALPHA    = 0.18

# Pinch thresholds (3D, normalized by hand size)
TM_THRESHOLD       = 0.30   # Thumb + Middle
TR_THRESHOLD       = 0.30   # Thumb + Ring

# Tap / hold timing
TAP_MAX_DURATION   = 0.35
HOLD_MIN_DURATION  = 0.55

# Scroll
# How much the hand's y must move per frame to trigger scroll
SCROLL_DEADZONE    = 0.003  # normalized units — ignore tiny drift
SCROLL_SPEED       = 600    # pyautogui scroll units per second
SCROLL_MULTIPLIER  = 120    # scales hand velocity → scroll amount

ACTIVE_ZONE_LEFT   = 0.10
ACTIVE_ZONE_RIGHT  = 0.90
ACTIVE_ZONE_TOP    = 0.10
ACTIVE_ZONE_BOTTOM = 0.90

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# ─────────────────────────────────────────────
# LANDMARK INDICES
# ─────────────────────────────────────────────
WRIST      = 0
THUMB_TIP  = 4;  THUMB_IP   = 3
INDEX_TIP  = 8;  INDEX_PIP  = 6
MIDDLE_TIP = 12; MIDDLE_PIP = 10; MIDDLE_MCP = 9
RING_TIP   = 16; RING_PIP   = 14
PINKY_TIP  = 20; PINKY_PIP  = 18


# ══════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading hand landmarker model (~25MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[INFO] Saved: {MODEL_PATH}")
    else:
        print(f"[INFO] Model found: {MODEL_PATH}")


# ══════════════════════════════════════════════
# COORDINATE MAPPER
# ══════════════════════════════════════════════
class CoordinateMapper:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()

    def map(self, nx, ny):
        cx = np.clip(nx, ACTIVE_ZONE_LEFT,  ACTIVE_ZONE_RIGHT)
        cy = np.clip(ny, ACTIVE_ZONE_TOP,   ACTIVE_ZONE_BOTTOM)
        sx = np.interp(cx, [ACTIVE_ZONE_LEFT,  ACTIVE_ZONE_RIGHT],  [0, self.screen_w])
        sy = np.interp(cy, [ACTIVE_ZONE_TOP,   ACTIVE_ZONE_BOTTOM], [0, self.screen_h])
        return int(sx), int(sy)


# ══════════════════════════════════════════════
# EMA SMOOTHER
# ══════════════════════════════════════════════
class EMASmoother:
    def __init__(self, alpha=SMOOTHING_ALPHA):
        self.alpha = alpha
        self.sx = self.sy = None

    def smooth(self, x, y):
        if self.sx is None:
            self.sx, self.sy = float(x), float(y)
        else:
            self.sx = self.alpha * x + (1 - self.alpha) * self.sx
            self.sy = self.alpha * y + (1 - self.alpha) * self.sy
        return int(self.sx), int(self.sy)

    def reset(self):
        self.sx = self.sy = None


# ══════════════════════════════════════════════
# GESTURE CLASSIFIER
# Returns a raw label each frame.
# ══════════════════════════════════════════════
class GestureClassifier:

    def hand_scale(self, lm):
        dx = lm[WRIST].x - lm[MIDDLE_MCP].x
        dy = lm[WRIST].y - lm[MIDDLE_MCP].y
        dz = lm[WRIST].z - lm[MIDDLE_MCP].z
        return max(np.sqrt(dx*dx + dy*dy + dz*dz), 1e-4)

    def d3(self, lm, a, b, scale):
        dx = lm[a].x - lm[b].x
        dy = lm[a].y - lm[b].y
        dz = lm[a].z - lm[b].z
        return np.sqrt(dx*dx + dy*dy + dz*dz) / scale

    def up(self, lm, tip, pip):
        return lm[tip].y < lm[pip].y

    def classify(self, lm):
        """
        Returns one of:
          'move'        - index only up
          'tm_pinch'    - thumb + middle close
          'tr_pinch'    - thumb + ring close
          'scroll'      - index + middle up (y-movement handled externally)
          'freeze'      - anything else
        """
        scale = self.hand_scale(lm)

        i_up = self.up(lm, INDEX_TIP,  INDEX_PIP)
        m_up = self.up(lm, MIDDLE_TIP, MIDDLE_PIP)
        r_up = self.up(lm, RING_TIP,   RING_PIP)
        p_up = self.up(lm, PINKY_TIP,  PINKY_PIP)

        d_tm = self.d3(lm, THUMB_TIP, MIDDLE_TIP, scale)
        d_tr = self.d3(lm, THUMB_TIP, RING_TIP,   scale)

        # ── Thumb + Middle pinch ──
        if d_tm < TM_THRESHOLD:
            return 'tm_pinch'

        # ── Thumb + Ring pinch ──
        if d_tr < TR_THRESHOLD:
            return 'tr_pinch'

        # ── Scroll: index + middle up, ring + pinky down ──
        if i_up and m_up and not r_up and not p_up:
            return 'scroll'

        # ── Move: index only up ──
        if i_up and not m_up and not r_up and not p_up:
            return 'move'

        return 'freeze'


# ══════════════════════════════════════════════
# GESTURE TIMER
# Handles tap vs hold for pinch gestures.
#
# For 'move', 'scroll', 'freeze' — emits directly.
# For pinch gestures:
#   start  → PENDING
#   <0.35s released → 'tap_X'
#   >0.55s held     → 'hold_X'
#   released after hold → 'hold_end_X'
# ══════════════════════════════════════════════
class GestureTimer:
    PINCH_GESTURES = ('tm_pinch', 'tr_pinch')

    def __init__(self):
        self.active   = None
        self.start    = None
        self.held     = False   # hold event already fired?

    def update(self, gesture):
        events = []
        now    = time.time()

        if gesture in ('move', 'scroll', 'freeze') or gesture is None:
            # Flush any pending pinch
            if self.active:
                elapsed = now - self.start
                if self.held:
                    events.append(f'hold_end_{self.active}')
                elif elapsed <= TAP_MAX_DURATION:
                    events.append(f'tap_{self.active}')
                # between tap and hold threshold: ambiguous, discard
                self.active = self.start = None
                self.held   = False
            if gesture:
                events.append(gesture)
            return events

        if gesture in self.PINCH_GESTURES:
            if self.active != gesture:
                # New pinch started
                if self.active and self.held:
                    events.append(f'hold_end_{self.active}')
                self.active = gesture
                self.start  = now
                self.held   = False
            else:
                # Same pinch continuing
                elapsed = now - self.start
                if not self.held and elapsed >= HOLD_MIN_DURATION:
                    self.held = True
                    events.append(f'hold_{gesture}')
            return events

        return events


# ══════════════════════════════════════════════
# SCROLL ENGINE
# Tracks hand y-position while 'scroll' gesture
# is active. Fires continuous scroll based on
# velocity (how fast hand moves up/down).
# ══════════════════════════════════════════════
class ScrollEngine:
    def __init__(self):
        self.prev_y   = None
        self.last_t   = None

    def update(self, current_y):
        """
        Call every frame when scroll gesture is active.
        Returns scroll amount (positive=up, negative=down).
        Proportional to hand velocity.
        """
        now = time.time()
        if self.prev_y is None:
            self.prev_y = current_y
            self.last_t = now
            return 0

        dt    = max(now - self.last_t, 1e-4)
        dy    = current_y - self.prev_y   # positive = hand moved down
        vel   = dy / dt                   # normalized units/sec

        self.prev_y = current_y
        self.last_t = now

        # Dead zone: ignore tiny drift
        if abs(dy) < SCROLL_DEADZONE:
            return 0

        # Negate: hand moves down (dy>0) → scroll down (negative)
        scroll_amount = -vel * SCROLL_MULTIPLIER
        return int(scroll_amount)

    def reset(self):
        self.prev_y = None
        self.last_t = None


# ══════════════════════════════════════════════
# MAIN CONTROLLER
# ══════════════════════════════════════════════
class HandMouseController:
    def __init__(self):
        ensure_model()

        self.mapper    = CoordinateMapper()
        self.smoother  = EMASmoother()
        self.classifier= GestureClassifier()
        self.timer     = GestureTimer()
        self.scroller  = ScrollEngine()

        self.latest_result = None
        self.frame_ts      = 0

        self.last_sx = self.mapper.screen_w // 2
        self.last_sy = self.mapper.screen_h // 2

        # HUD state
        self.hud_gesture = 'freeze'
        self.hud_event   = ''

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.75,
            min_hand_presence_confidence=0.75,
            min_tracking_confidence=0.75,
            result_callback=self._on_result,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera.")

        self.cam_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[INFO] Camera: {self.cam_w}x{self.cam_h} | "
              f"Screen: {self.mapper.screen_w}x{self.mapper.screen_h}")
        print("[INFO] Gestures:")
        print("  Index only up              → move mouse")
        print("  Thumb+Middle tap           → left click")
        print("  Thumb+Middle hold          → right click")
        print("  Thumb+Ring tap             → (reserved)")
        print("  Thumb+Ring hold            → right click alt")
        print("  Index+Middle + move down   → scroll down")
        print("  Index+Middle + move up     → scroll up")
        print("  q                          → quit")

    def _on_result(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def _execute(self, events, lm):
        for ev in events:

            # ── MOVE ──
            if ev == 'move':
                sx, sy = self.mapper.map(lm[INDEX_TIP].x, lm[INDEX_TIP].y)
                sx, sy = self.smoother.smooth(sx, sy)
                self.last_sx, self.last_sy = sx, sy
                pyautogui.moveTo(sx, sy)
                self.hud_gesture = 'move'
                self.scroller.reset()

            # ── SCROLL (continuous, velocity-based) ──
            elif ev == 'scroll':
                # Use wrist y as anchor (stable center of hand)
                hand_y = lm[WRIST].y
                amt    = self.scroller.update(hand_y)
                if amt != 0:
                    pyautogui.scroll(amt)
                self.hud_gesture = f'scroll {"↑" if amt > 0 else "↓" if amt < 0 else "-"}'

            # ── FREEZE ──
            elif ev == 'freeze':
                self.scroller.reset()
                self.hud_gesture = 'freeze'

            # ── LEFT CLICK: thumb+middle tap ──
            elif ev == 'tap_tm_pinch':
                pyautogui.click(self.last_sx, self.last_sy)
                self.hud_event = 'LEFT CLICK'
                print("[LC] Left click")

            # ── RIGHT CLICK: thumb+middle hold ──
            elif ev == 'hold_tm_pinch':
                pyautogui.rightClick(self.last_sx, self.last_sy)
                self.hud_event = 'RIGHT CLICK'
                print("[RC] Right click")

            elif ev == 'hold_end_tm_pinch':
                self.hud_event = ''

            # ── THUMB+RING tap: reserved / right click alt ──
            elif ev == 'tap_tr_pinch':
                pyautogui.rightClick(self.last_sx, self.last_sy)
                self.hud_event = 'RIGHT CLICK'
                print("[RC] Right click (tr)")

            elif ev == 'hold_tr_pinch':
                self.hud_event = ''

            elif ev == 'hold_end_tr_pinch':
                self.hud_event = ''

            # pinch gestures in progress (no event yet)
            elif ev in ('tm_pinch', 'tr_pinch'):
                self.hud_gesture = ev
                self.scroller.reset()

    def _draw_zone(self, frame):
        x1 = int(ACTIVE_ZONE_LEFT   * self.cam_w)
        x2 = int(ACTIVE_ZONE_RIGHT  * self.cam_w)
        y1 = int(ACTIVE_ZONE_TOP    * self.cam_h)
        y2 = int(ACTIVE_ZONE_BOTTOM * self.cam_h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 80), 1)

    def _draw_landmarks(self, frame, lm):
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),
        ]
        pts = [(int(l.x * self.cam_w), int(l.y * self.cam_h)) for l in lm]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], (50, 140, 50), 2)
        for px, py in pts:
            cv2.circle(frame, (px, py), 3, (0, 180, 140), -1)
        highlights = {
            INDEX_TIP:  (0,   80, 255),
            MIDDLE_TIP: (0,  180, 255),
            RING_TIP:   (180,  0, 255),
            THUMB_TIP:  (255, 160,  0),
        }
        for idx, col in highlights.items():
            cv2.circle(frame, pts[idx], 8, col, -1)

    def _draw_hud(self, frame):
        color_map = {
            'move':       (200, 200, 200),
            'freeze':     (70,   70,  70),
            'tm_pinch':   (0,   255, 120),
            'tr_pinch':   (180,   0, 255),
        }
        col = (100, 220, 100)
        for k, v in color_map.items():
            if self.hud_gesture.startswith(k):
                col = v
                break

        cv2.putText(frame, f"Gesture: {self.hud_gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
        cv2.putText(frame, f"Mouse: ({self.last_sx},{self.last_sy})", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 130, 130), 1)
        if self.hud_event:
            cv2.putText(frame, self.hud_event, (10, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Frame read failed.")
                break

            frame = cv2.flip(frame, 1)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            self.frame_ts += 1
            self.landmarker.detect_async(mp_image, self.frame_ts)
            self._draw_zone(frame)

            result = self.latest_result
            if result and result.hand_landmarks:
                lm     = result.hand_landmarks[0]
                raw    = self.classifier.classify(lm)
                events = self.timer.update(raw)

                self._draw_landmarks(frame, lm)
                self._execute(events, lm)

            else:
                self.timer.update(None)
                self.scroller.reset()
                self.smoother.reset()
                self.hud_gesture = 'no hand'
                cv2.putText(frame, "No hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)

            self._draw_hud(frame)
            cv2.imshow("Hand Mouse v4 | q=quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.landmarker.close()
        cv2.destroyAllWindows()
        print("[INFO] Exited.")


if __name__ == "__main__":
    controller = HandMouseController()
    controller.run()
"""
Hand Gesture Mouse Controller v6
==================================
MediaPipe Tasks API (0.10.30+)

Gesture map:
  Index only up                         → move mouse
  Thumb + Middle tap   (< 0.35s)        → left click
  Thumb + Middle hold  (> 0.55s)        → right click + mouse hold (drag/select)
  Thumb + Ring tap                      → right click (alt)
  Index + Middle up + hand in top zone  → scroll up (continuous)
  Index + Middle up + hand in bot zone  → scroll down (continuous)
  Index + Middle up + hand in mid zone  → scroll idle (stop)
  Anything else                         → freeze mouse
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
pyautogui.PAUSE    = 0

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────

SMOOTHING_ALPHA    = 0.18

# Pinch thresholds (3D, normalized by hand size)
TM_THRESHOLD       = 0.30    # Thumb + Middle
TR_THRESHOLD       = 0.30    # Thumb + Ring

# Tap / hold timing
TAP_MAX_DURATION   = 0.35    # sec — released before this = tap
HOLD_MIN_DURATION  = 0.55    # sec — held past this = hold

# Active zone — asymmetric to match real hand movement range.
# Your hand physically can't reach the absolute camera edges,
# so we shrink the mapped zone slightly on all sides EXCEPT
# the bottom (taskbar needs to be reachable).
# Tweak these if cursor still doesn't reach an edge:
#   - Can't reach left  → lower  ACTIVE_ZONE_LEFT
#   - Can't reach right → raise  ACTIVE_ZONE_RIGHT
#   - Can't reach top   → lower  ACTIVE_ZONE_TOP
#   - Can't reach bottom→ raise  ACTIVE_ZONE_BOTTOM
ACTIVE_ZONE_LEFT   = 0.08
ACTIVE_ZONE_RIGHT  = 0.92
ACTIVE_ZONE_TOP    = 0.08
ACTIVE_ZONE_BOTTOM = 0.97

# Velocity-based scroll
# Scroll only fires when fingers are moving. Stopped = no scroll.
SCROLL_SENSITIVITY = 60      # raise = faster, lower = slower
SCROLL_DEADZONE    = 0.004   # min Y delta per frame to register motion


MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

# ─────────────────────────────────────────────────────
# LANDMARK INDICES
# ─────────────────────────────────────────────────────
WRIST      = 0
THUMB_TIP  = 4;  THUMB_IP   = 3
INDEX_TIP  = 8;  INDEX_PIP  = 6
MIDDLE_TIP = 12; MIDDLE_PIP = 10; MIDDLE_MCP = 9
RING_TIP   = 16; RING_PIP   = 14
PINKY_TIP  = 20; PINKY_PIP  = 18


# ══════════════════════════════════════════════════════
# MODEL DOWNLOADER
# ══════════════════════════════════════════════════════
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading hand landmarker model (~25MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[INFO] Saved: {MODEL_PATH}")
    else:
        print(f"[INFO] Model found: {MODEL_PATH}")


# ══════════════════════════════════════════════════════
# COORDINATE MAPPER
# Maps normalized camera coords → screen pixels.
# Full 0.0–1.0 active zone so taskbar is reachable.
# ══════════════════════════════════════════════════════
class CoordinateMapper:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()

    def map(self, nx, ny):
        cx = np.clip(nx, ACTIVE_ZONE_LEFT,  ACTIVE_ZONE_RIGHT)
        cy = np.clip(ny, ACTIVE_ZONE_TOP,   ACTIVE_ZONE_BOTTOM)
        sx = np.interp(cx, [ACTIVE_ZONE_LEFT,  ACTIVE_ZONE_RIGHT],
                           [0, self.screen_w])
        sy = np.interp(cy, [ACTIVE_ZONE_TOP,   ACTIVE_ZONE_BOTTOM],
                           [0, self.screen_h])
        return int(sx), int(sy)


# ══════════════════════════════════════════════════════
# EMA SMOOTHER
# ══════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════
# VELOCITY SCROLL ENGINE
#
# Fires scroll only when fingers are actively moving.
# Stopped fingers = zero scroll, no dead zone needed.
#
# How it works:
#   - Tracks index tip Y each frame
#   - delta = current_y - previous_y
#   - abs(delta) < SCROLL_DEADZONE → ignore (hand tremor)
#   - scroll amount = -delta * SCROLL_SENSITIVITY
#     finger moves down (delta>0) → scroll down (negative)
#     finger moves up   (delta<0) → scroll up   (positive)
# ══════════════════════════════════════════════════════
class VelocityScrollEngine:
    def __init__(self):
        self.prev_y    = None
        self.direction = 0
        self.speed     = 0

    def update(self, finger_y):
        if self.prev_y is None:
            self.prev_y = finger_y
            return 0

        delta       = finger_y - self.prev_y
        self.prev_y = finger_y

        if abs(delta) < SCROLL_DEADZONE:
            self.direction = 0
            self.speed     = 0
            return 0

        amount         = -delta * SCROLL_SENSITIVITY
        self.direction = 1 if amount > 0 else -1
        self.speed     = abs(int(amount))
        return int(amount)

    def reset(self):
        self.prev_y    = None
        self.direction = 0
        self.speed     = 0


# ══════════════════════════════════════════════════════
# GESTURE CLASSIFIER
# Returns a raw label each frame.
# ══════════════════════════════════════════════════════
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
        Priority order (high → low):
          tm_pinch  > tr_pinch  > im_close > scroll > move > freeze
        """
        scale = self.hand_scale(lm)

        i_up = self.up(lm, INDEX_TIP,  INDEX_PIP)
        m_up = self.up(lm, MIDDLE_TIP, MIDDLE_PIP)
        r_up = self.up(lm, RING_TIP,   RING_PIP)
        p_up = self.up(lm, PINKY_TIP,  PINKY_PIP)

        d_tm = self.d3(lm, THUMB_TIP,  MIDDLE_TIP, scale)
        d_tr = self.d3(lm, THUMB_TIP, RING_TIP, scale)

        # Thumb+Middle pinch → left click (tap) / mouse hold (hold)
        if d_tm < TM_THRESHOLD:
            return 'tm_pinch'

        # Thumb+Ring pinch → right click
        if d_tr < TR_THRESHOLD:
            return 'tr_pinch'

        # Scroll: index + middle up, ring + pinky down
        if i_up and m_up and not r_up and not p_up:
            return 'scroll'

        # Move: index only up
        if i_up and not m_up and not r_up and not p_up:
            return 'move'

        return 'freeze'


# ══════════════════════════════════════════════════════
# GESTURE TIMER
# Converts per-frame raw labels into timed events.
#
# Stateless gestures (move, scroll, freeze):
#   emitted directly every frame.
#
# Pinch gestures (tm_pinch, tr_pinch):
#   PENDING  → tap fired on release if < TAP_MAX
#   PENDING  → hold fired if sustained > HOLD_MIN
#   HOLDING  → hold_end fired on release
# ══════════════════════════════════════════════════════
class GestureTimer:
    TIMED = ('tm_pinch', 'tr_pinch')

    def __init__(self):
        self.active = None
        self.start  = None
        self.held   = False

    def update(self, gesture):
        events = []
        now    = time.time()

        if gesture not in self.TIMED:
            # Flush pending timed gesture
            if self.active:
                elapsed = now - self.start
                if self.held:
                    events.append(f'hold_end_{self.active}')
                elif elapsed <= TAP_MAX_DURATION:
                    events.append(f'tap_{self.active}')
                self.active = self.start = None
                self.held   = False
            if gesture:
                events.append(gesture)
            return events

        # Timed gesture
        if self.active != gesture:
            if self.active and self.held:
                events.append(f'hold_end_{self.active}')
            self.active = gesture
            self.start  = now
            self.held   = False
        else:
            elapsed = now - self.start
            if not self.held and elapsed >= HOLD_MIN_DURATION:
                self.held = True
                events.append(f'hold_{gesture}')

        return events


# ══════════════════════════════════════════════════════
# MAIN CONTROLLER
# ══════════════════════════════════════════════════════
class HandMouseController:
    def __init__(self):
        ensure_model()

        self.mapper    = CoordinateMapper()
        self.smoother  = EMASmoother()
        self.classifier= GestureClassifier()
        self.timer     = GestureTimer()
        self.scroller  = VelocityScrollEngine()

        self.latest_result  = None
        self.frame_ts       = 0

        self.last_sx        = self.mapper.screen_w // 2
        self.last_sy        = self.mapper.screen_h // 2

        self.mouse_held     = False   # True while im_close hold active

        self.hud_gesture    = 'freeze'
        self.hud_event      = ''
        self.hud_event_time = 0.0

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
        print("[INFO] Gesture map:")
        print("  Index only               → move")
        print("  Thumb+Middle tap         → left click")
        print("  Thumb+Middle hold        → mouse hold (drag/select)")
        print("  Thumb+Ring tap           → right click")
        print("  Index+Middle + top zone  → scroll up")
        print("  Index+Middle + bot zone  → scroll down")
        print("  Index+Middle + mid zone  → scroll stop")
        print("  q                        → quit")

    def _on_result(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def _set_event(self, label):
        self.hud_event      = label
        self.hud_event_time = time.time()

    def _release_mouse_hold(self):
        if self.mouse_held:
            pyautogui.mouseUp()
            self.mouse_held = False
            self._set_event('HOLD RELEASED')
            print("[MH] Mouse hold released")

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

            # ── EDGE SCROLL ──
            elif ev == 'scroll':
                # Use index tip Y — stays in frame better than wrist
                amt = self.scroller.update(lm[INDEX_TIP].y)
                if amt != 0:
                    pyautogui.scroll(amt)
                direction = self.scroller.direction
                arrow = '↑' if direction > 0 else ('↓' if direction < 0 else '—')
                self.hud_gesture = f'scroll {arrow}  spd:{self.scroller.speed}'

            # ── FREEZE ──
            elif ev == 'freeze':
                self.scroller.reset()
                self.hud_gesture = 'freeze'

            # ── LEFT CLICK: thumb+middle tap ──
            elif ev == 'tap_tm_pinch':
                pyautogui.click(self.last_sx, self.last_sy)
                self._set_event('LEFT CLICK')
                self.hud_gesture = 'tm_pinch'
                print("[LC] Left click")

            # ── MOUSE HOLD START: thumb+middle hold ──
            elif ev == 'hold_tm_pinch':
                if not self.mouse_held:
                    pyautogui.mouseDown(self.last_sx, self.last_sy)
                    self.mouse_held = True
                    self._set_event('MOUSE HOLD')
                    self.hud_gesture = 'tm_pinch (HOLD)'
                    print("[MH] Mouse hold start")

            # ── MOUSE HOLD END: thumb+middle released after hold ──
            elif ev == 'hold_end_tm_pinch':
                self._release_mouse_hold()
                self.hud_gesture = 'freeze'

            # ── RIGHT CLICK: thumb+ring tap ──
            elif ev == 'tap_tr_pinch':
                pyautogui.rightClick(self.last_sx, self.last_sy)
                self._set_event('RIGHT CLICK')
                self.hud_gesture = 'tr_pinch'
                print("[RC] Right click")

            elif ev in ('hold_tr_pinch', 'hold_end_tr_pinch'):
                self.hud_gesture = 'freeze'

            # In-progress timed gestures (no event yet — just update HUD)
            elif ev in ('tm_pinch', 'tr_pinch'):
                self.hud_gesture = ev
                self.scroller.reset()

        # While mouse is held, follow index tip for drag
        if self.mouse_held and lm is not None:
            sx, sy = self.mapper.map(lm[INDEX_TIP].x, lm[INDEX_TIP].y)
            sx, sy = self.smoother.smooth(sx, sy)
            self.last_sx, self.last_sy = sx, sy
            pyautogui.moveTo(sx, sy)


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
            cv2.line(frame, pts[a], pts[b], (40, 120, 40), 2)
        for px, py in pts:
            cv2.circle(frame, (px, py), 3, (0, 170, 130), -1)
        highlights = {
            INDEX_TIP:  (0,   80, 255),
            MIDDLE_TIP: (0,  180, 255),
            RING_TIP:   (180,  0, 255),
            THUMB_TIP:  (255, 160,   0),
        }
        for idx, col in highlights.items():
            cv2.circle(frame, pts[idx], 8, col, -1)

    def _draw_hud(self, frame):
        # Expire event label after 0.8s
        if time.time() - self.hud_event_time > 0.8:
            self.hud_event = ''

        color_map = {
            'move':      (200, 200, 200),
            'freeze':    (70,   70,  70),
            'tm_pinch':  (0,   255, 120),
            'tr_pinch':  (180,   0, 255),
            'im_close':  (255, 180,   0),
            'scroll':    (0,   200, 255),
            'no hand':   (0,     0, 200),
        }
        col = (100, 200, 100)
        for k, v in color_map.items():
            if self.hud_gesture.startswith(k):
                col = v
                break

        cv2.putText(frame, f"Gesture: {self.hud_gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.putText(frame, f"Mouse: ({self.last_sx},{self.last_sy})", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 120, 120), 1)
        if self.hud_event:
            cv2.putText(frame, self.hud_event, (10, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 180), 2)
        if self.mouse_held:
            cv2.putText(frame, "[ HOLDING ]", (10, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Frame read failed.")
                break

            frame    = cv2.flip(frame, 1)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            self.frame_ts += 1
            self.landmarker.detect_async(mp_image, self.frame_ts)

            result = self.latest_result
            lm     = None

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
                self._release_mouse_hold()
                self.hud_gesture = 'no hand'

            self._draw_hud(frame)
            cv2.imshow("Hand Mouse v5 | q=quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self._release_mouse_hold()
        self.cap.release()
        self.landmarker.close()
        cv2.destroyAllWindows()
        print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    controller = HandMouseController()
    controller.run()
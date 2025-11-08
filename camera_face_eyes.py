"""
camera_face_eyes.py
Protótipo final -> detecta rosto, olhos, cabeça (head-pose), e alerta se dorme ou perde foco.
Requisitos: Python 3.12, mediapipe, opencv-python, numpy, pygame (opcional para MP3)
Rode: python camera_face_eyes.py
Aperte 'q' para sair.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import csv
from datetime import datetime
import winsound
import ctypes
from ctypes import wintypes
import math
import os

# -------------------- WinAPI helpers (ctypes) --------------------
user32 = ctypes.WinDLL('user32', use_last_error=True)
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

SW_MINIMIZE = 6
SW_RESTORE = 9
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_SHOWWINDOW = 0x0040

SetWindowPos = user32.SetWindowPos
SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
SetWindowPos.restype = wintypes.BOOL

ShowWindow = user32.ShowWindow
ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
ShowWindow.restype = wintypes.BOOL

# WNDENUMPROC callback type (correct approach)
WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
EnumWindows = user32.EnumWindows
EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
EnumWindows.restype = wintypes.BOOL

GetWindowText = user32.GetWindowTextW
GetWindowText.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
GetWindowText.restype = ctypes.c_int

GetWindowTextLength = user32.GetWindowTextLengthW
GetWindowTextLength.argtypes = [wintypes.HWND]
GetWindowTextLength.restype = ctypes.c_int

IsWindowVisible = user32.IsWindowVisible
IsWindowVisible.argtypes = [wintypes.HWND]
IsWindowVisible.restype = wintypes.BOOL

GetForegroundWindow = user32.GetForegroundWindow
GetForegroundWindow.restype = wintypes.HWND

# LASTINPUTINFO structure for idle detection
class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [('cbSize', ctypes.c_uint), ('dwTime', ctypes.c_uint)]

def get_idle_seconds():
    """Retorna segundos desde o último input (mouse/teclado) no Windows."""
    lii = LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
    if not user32.GetLastInputInfo(ctypes.byref(lii)):
        return 0.0
    millis = kernel32.GetTickCount() - lii.dwTime
    return millis / 1000.0

def get_foreground_window_title():
    """Retorna o título da janela em primeiro plano (string)."""
    try:
        hwnd = GetForegroundWindow()
        length = GetWindowTextLength(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        GetWindowText(hwnd, buff, length + 1)
        return buff.value
    except Exception:
        return ""

def set_window_topmost_by_name(win_name, top=True):
    """Procura janela por título exato e seta topmost ou notopmost."""
    hwnd = user32.FindWindowW(None, win_name)
    if hwnd:
        SetWindowPos(hwnd, HWND_TOPMOST if top else HWND_NOTOPMOST, 0,0,0,0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
        return True
    return False

def minimize_all_other_windows(except_hwnd=None, exclude_title_contains=None):
    """Minimiza janelas visíveis exceto a nossa (passar exceção) e títulos contendo exclude_title_contains."""
    def callback(hwnd, lParam):
        try:
            if not IsWindowVisible(hwnd):
                return True
            length = GetWindowTextLength(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            GetWindowText(hwnd, buff, length + 1)
            title = buff.value or ""
            if not title:
                return True
            if except_hwnd and hwnd == except_hwnd:
                return True
            if exclude_title_contains and exclude_title_contains.lower() in title.lower():
                return True
            ShowWindow(hwnd, SW_MINIMIZE)
        except Exception:
            pass
        return True
    enum_proc = WNDENUMPROC(callback)
    EnumWindows(enum_proc, 0)

# -------------------- Configurações --------------------
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.20       # abaixo disso consideramos olho "fechado"
CONSEC_FRAMES = 3          # frames consecutivos para confirmar "fechado"
SMOOTH_WINDOW = 5          # média móvel de EAR em N frames
LOG_CSV = "eye_log.csv"

# head-pose thresholds
PITCH_CLOSE_THRESHOLD = 12.0
YAW_CLOSE_THRESHOLD = 25.0
EYE_CLOSED_CONSEC_FRAMES = CONSEC_FRAMES

# alert behavior
ALERT_SECONDS = 15
MINIMIZE_WINDOWS_ON_ALERT = True
ALERT_AUDIO_FILE = "alert.mp3"   # opcional
ALERT_IMAGE_FILE = "alert.png"   # opcional

# phone detection timing
phone_start_time = None
PHONE_ALERT_SECONDS = 8   # segundos antes de alertar "no celular"

# -------------------- Inicializações --------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# tenta carregar pygame (para MP3). se não disponível, fallback para winsound Beep
try:
    import pygame
    pygame.mixer.pre_init(44100, -16, 2, 2048)
    pygame.init()
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

# tenta carregar imagem de alerta (opcional)
_alert_img = None
if os.path.exists(ALERT_IMAGE_FILE):
    try:
        _tmp = cv2.imread(ALERT_IMAGE_FILE, cv2.IMREAD_UNCHANGED)
        if _tmp is not None:
            _alert_img = _tmp
    except Exception:
        _alert_img = None

# inicializa câmera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Erro: não consegui abrir a câmera. Verifique índice ou drivers.")

ear_window = deque(maxlen=SMOOTH_WINDOW)
consec_closed = 0

# alerta control vars
closed_start_time = None
alert_played = False

# cria CSV (cabeçalho)
with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "ear", "pitch", "yaw", "is_phone", "eyes_open"])

# -------------------- Funções utilitárias --------------------
def eye_aspect_ratio(eye_points):
    p1, p2, p3, p4, p5, p6 = eye_points
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    ear = (A + B) / (2.0 * (C + 1e-8))
    return ear

# solvePnP model points (approx)
PNP_IDX = [1, 152, 33, 263, 61, 291]
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1)
], dtype=np.float32)

def estimate_head_pose(landmarks, img_w, img_h):
    image_points = np.array([landmarks[i] for i in PNP_IDX], dtype=np.float32)
    focal_length = img_w
    center = (img_w/2, img_h/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4,1))
    success, rotation_vec, translation_vec = cv2.solvePnP(MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rotation_vec)
    sy = math.sqrt(rmat[0,0]*rmat[0,0] + rmat[1,0]*rmat[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rmat[2,1], rmat[2,2])
        y = math.atan2(-rmat[2,0], sy)
        z = math.atan2(rmat[1,0], rmat[0,0])
    else:
        x = math.atan2(-rmat[1,2], rmat[1,1])
        y = math.atan2(-rmat[2,0], sy)
        z = 0
    pitch = math.degrees(x)
    yaw = math.degrees(y)
    roll = math.degrees(z)
    return yaw, pitch, roll

def detect_not_focused(landmarks, w, h, ear_smoothed):
    score = 0.0
    reasons = []
    yaw = pitch = roll = 0.0
    try:
        yaw, pitch, roll = estimate_head_pose(landmarks, w, h)
        if abs(yaw) > 20:
            score += 0.4; reasons.append(f"yaw={yaw:.1f}")
        if pitch > 15:
            score += 0.4; reasons.append(f"pitch={pitch:.1f}")
    except Exception:
        pass
    try:
        left_eye_c = np.mean([landmarks[i] for i in LEFT_EYE_IDX], axis=0)
        right_eye_c = np.mean([landmarks[i] for i in RIGHT_EYE_IDX], axis=0)
        eyes_center = (left_eye_c + right_eye_c) / 2.0
        nose = landmarks[1]
        dx = (nose[0] - eyes_center[0]) / w
        dy = (nose[1] - eyes_center[1]) / h
        if abs(dx) > 0.12:
            score += 0.15; reasons.append(f"nose_dx={dx:.2f}")
        if dy > 0.08:
            score += 0.15; reasons.append(f"nose_dy={dy:.2f}")
    except Exception:
        pass
    idle_sec = get_idle_seconds()
    if idle_sec > 5:
        score += 0.1; reasons.append(f"idle={int(idle_sec)}s")
    fg = get_foreground_window_title()
    main_title = "Face+Eyes Detector"
    if fg and main_title.lower() not in fg.lower():
        score += 0.1; reasons.append(f"fg='{fg[:30]}'")
    if ear_smoothed is not None and pitch > 12 and ear_smoothed > 0.18:
        score += 0.2; reasons.append("open_eyes_head_down")
    score = min(1.0, score)
    is_phone = score >= 0.4
    return is_phone, score, ";".join(reasons), yaw, pitch

def play_alert_audio_once():
    global alert_played
    if alert_played:
        return
    try:
        if PYGAME_AVAILABLE and os.path.exists(ALERT_AUDIO_FILE):
            pygame.mixer.music.load(ALERT_AUDIO_FILE)
            pygame.mixer.music.play()
        else:
            winsound.Beep(1000, 700)
    except Exception:
        try:
            winsound.Beep(1000, 700)
        except Exception:
            pass
    alert_played = True

# -------------------- Loop principal --------------------
try:
    # variables for phone alert timing
    phone_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar frame. Saindo.")
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ear_val = None
        eyes_open = False
        pitch = 0.0
        yaw = 0.0
        is_phone = False
        phone_score = 0.0

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            landmarks = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)

            left_eye_pts = landmarks[LEFT_EYE_IDX]
            right_eye_pts = landmarks[RIGHT_EYE_IDX]

            left_ear = eye_aspect_ratio(left_eye_pts)
            right_ear = eye_aspect_ratio(right_eye_pts)
            ear_val = float((left_ear + right_ear) / 2.0)

            ear_window.append(ear_val)
            ear_smoothed = float(np.mean(ear_window))

            # head pose for decision
            yaw, pitch, roll = estimate_head_pose(landmarks, w, h)

            # decide real eye-closed vs head-down (phone)
            is_ear_low = ear_smoothed < EAR_THRESHOLD
            is_head_down = pitch > PITCH_CLOSE_THRESHOLD
            is_head_turned = abs(yaw) > YAW_CLOSE_THRESHOLD
            is_real_eye_closed = is_ear_low and (not is_head_down) and (not is_head_turned)

            if is_real_eye_closed:
                consec_closed += 1
            else:
                consec_closed = 0

            eyes_open = not (consec_closed >= EYE_CLOSED_CONSEC_FRAMES)

            # detect phone/focus loss
            is_phone, phone_score, phone_reasons, yaw_calc, pitch_calc = detect_not_focused(landmarks, w, h, ear_smoothed)

            # draw indicators for phone/focus
            if is_phone:
                cv2.putText(frame, f"Desfocado: ({phone_score:.2f})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
            else:
                cv2.putText(frame, f"FOCO OK ({phone_score:.2f})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

            # phone alert timing
            now = time.time()
            if is_phone:
                if phone_start_time is None:
                    phone_start_time = now
                if (now - phone_start_time) >= PHONE_ALERT_SECONDS:
                    # for phone alert we use the same alert function (you can separate sounds)
                    play_alert_audio_once()
            else:
                phone_start_time = None

            # draw eyes and EAR
            for (x, y) in np.vstack([left_eye_pts, right_eye_pts]).astype(int):
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            status_text = f"EAR: {ear_smoothed:.2f}  Open: {eyes_open}"
            color = (0, 255, 0) if eyes_open else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # face bounding box
            x_coords = [int(p[0]) for p in landmarks]
            y_coords = [int(p[1]) for p in landmarks]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

            # ---------- Dormir/inativo detection (tempo real) ----------
            if not eyes_open:
                if closed_start_time is None:
                    closed_start_time = time.time()
                    alert_played = False
                elapsed = time.time() - closed_start_time
                cv2.putText(frame, f"Inativo: {int(elapsed)}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                if elapsed >= ALERT_SECONDS and not alert_played:
                    # minimize windows if configured
                    if MINIMIZE_WINDOWS_ON_ALERT:
                        try:
                            my_hwnd = GetForegroundWindow()
                            minimize_all_other_windows(except_hwnd=my_hwnd, exclude_title_contains=None)
                        except Exception:
                            pass

                    # play audio once
                    play_alert_audio_once()

                    # create alert fullscreen window (topmost) and blink until eyes open
                    alert_win = "ALERTA - FOQUE!"
                    cv2.namedWindow(alert_win, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(alert_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    for _ in range(8):
                        set_window_topmost_by_name(alert_win, top=True)
                        time.sleep(0.03)

                    flash_on = True
                    flash_timer = time.time()

                    # alert loop: show blinking overlay + small camera, stop audio when eyes re-open
                    while True:
                        ret2, f2 = cap.read()
                        if not ret2:
                            break
                        rgb2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
                        res2 = face_mesh.process(rgb2)
                        ear_now = None
                        eyes_open_now = False
                        if res2.multi_face_landmarks:
                            lm2 = res2.multi_face_landmarks[0]
                            landmarks2 = np.array([[p.x * w, p.y * h] for p in lm2.landmark], dtype=np.float32)
                            lpts = landmarks2[LEFT_EYE_IDX]; rpts = landmarks2[RIGHT_EYE_IDX]
                            le = eye_aspect_ratio(lpts); re = eye_aspect_ratio(rpts)
                            ear_now = (le + re) / 2.0
                            if ear_now >= EAR_THRESHOLD:
                                eyes_open_now = True

                        disp = f2.copy()
                        h2, w2 = disp.shape[:2]
                        overlay = disp.copy()
                        alpha = 0.6 if flash_on else 0.25
                        cv2.rectangle(overlay, (0,0), (w2, h2), (0,0,255), -1)
                        cv2.addWeighted(overlay, alpha, disp, 1 - alpha, 0, disp)

                        cv2.putText(disp, "ACORDA!", (int(w2*0.05), int(h2*0.45)),
                                    cv2.FONT_HERSHEY_DUPLEX, 2.0, (255,255,255), 4, cv2.LINE_AA)

                        # small camera preview
                        small = cv2.resize(f2, (int(w2*0.25), int(h2*0.25)))
                        disp[10:10+small.shape[0], 10:10+small.shape[1]] = small

                        # optional alert image on the right
                        if _alert_img is not None:
                            try:
                                ih, iw = _alert_img.shape[:2]
                                target_w = int(w2 * 0.30)
                                scale = target_w / max(1, iw)
                                target_h = int(ih * scale)
                                resized = cv2.resize(_alert_img, (target_w, target_h))
                                x_off = w2 - target_w - 40
                                y_off = int(h2 * 0.25)
                                if resized.shape[2] == 4:
                                    alpha_chan = resized[:, :, 3] / 255.0
                                    for c in range(3):
                                        disp[y_off:y_off+target_h, x_off:x_off+target_w, c] = (
                                            alpha_chan * resized[:, :, c] + (1 - alpha_chan) * disp[y_off:y_off+target_h, x_off:x_off+target_w, c]
                                        ).astype(disp.dtype)
                                else:
                                    disp[y_off:y_off+target_h, x_off:x_off+target_w] = resized
                            except Exception:
                                pass

                        cv2.imshow(alert_win, disp)

                        if time.time() - flash_timer > 0.5:
                            flash_on = not flash_on
                            flash_timer = time.time()

                        # stop audio immediately when eyes reopen
                        if eyes_open_now:
                            try:
                                if PYGAME_AVAILABLE and pygame.mixer.get_init():
                                    pygame.mixer.music.stop()
                            except Exception:
                                pass
                            break

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            try:
                                if PYGAME_AVAILABLE and pygame.mixer.get_init():
                                    pygame.mixer.music.stop()
                            except Exception:
                                pass
                            break

                    # close alert window and reset
                    try:
                        cv2.destroyWindow(alert_win)
                        set_window_topmost_by_name(alert_win, top=False)
                    except Exception:
                        pass
                    closed_start_time = None
                    alert_played = False
            else:
                closed_start_time = None
                alert_played = False

        else:
            # no face found: reset some state
            ear_window.clear()
            consec_closed = 0
            closed_start_time = None
            alert_played = False
            cv2.putText(frame, "Rosto nao detectado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # main small window
        cv2.imshow("Face+Eyes Detector", frame)

        # logging for later calibration
        if ear_val is not None:
            try:
                with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.utcnow().isoformat(), f"{ear_val:.4f}", f"{pitch:.2f}", f"{yaw:.2f}", int(is_phone), int(eyes_open)])
            except Exception:
                pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    if PYGAME_AVAILABLE:
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except Exception:
            pass
    print("Encerrado. Logs em:", LOG_CSV)

"""
camera_face_eyes_hud.py
Versão com HUD melhorado e mais visual
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

# -------------------- WinAPI helpers --------------------
user32 = ctypes.WinDLL('user32', use_last_error=True)
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

SW_MINIMIZE = 6
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

WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
EnumWindows = user32.EnumWindows
GetWindowText = user32.GetWindowTextW
GetWindowTextLength = user32.GetWindowTextLengthW
IsWindowVisible = user32.IsWindowVisible
GetForegroundWindow = user32.GetForegroundWindow

class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [('cbSize', ctypes.c_uint), ('dwTime', ctypes.c_uint)]

def get_idle_seconds():
    lii = LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
    if not user32.GetLastInputInfo(ctypes.byref(lii)):
        return 0.0
    millis = kernel32.GetTickCount() - lii.dwTime
    return millis / 1000.0

def get_foreground_window_title():
    try:
        hwnd = GetForegroundWindow()
        length = GetWindowTextLength(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        GetWindowText(hwnd, buff, length + 1)
        return buff.value
    except Exception:
        return ""

def minimize_all_other_windows(except_hwnd=None, exclude_title_contains=None):
    def callback(hwnd, lParam):
        try:
            if not IsWindowVisible(hwnd):
                return True
            length = GetWindowTextLength(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            GetWindowText(hwnd, buff, length + 1)
            title = buff.value or ""
            if not title or (except_hwnd and hwnd == except_hwnd):
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
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 1
SMOOTH_WINDOW = 5
LOG_CSV = "eye_log.csv"

PITCH_CLOSE_THRESHOLD = 12.0
YAW_CLOSE_THRESHOLD = 25.0
EYE_CLOSED_CONSEC_FRAMES = CONSEC_FRAMES

ALERT_SECONDS = 10
PHONE_ALERT_SECONDS = 8
MINIMIZE_WINDOWS_ON_ALERT = True
ALERT_AUDIO_FILE = "alert.mp3"
ALERT_IMAGE_FILE = "Acorda.png"

# -------------------- HUD Config --------------------
HUD_BG_COLOR = (20, 20, 30)
HUD_ACCENT_OK = (0, 255, 150)
HUD_ACCENT_WARNING = (0, 165, 255)
HUD_ACCENT_DANGER = (0, 0, 255)

# -------------------- Inicializações --------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

try:
    import pygame
    pygame.mixer.pre_init(44100, -16, 2, 2048)
    pygame.init()
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

_alert_img = None
if os.path.exists(ALERT_IMAGE_FILE):
    try:
        _tmp = cv2.imread(ALERT_IMAGE_FILE, cv2.IMREAD_UNCHANGED)
        if _tmp is not None:
            _alert_img = _tmp
    except Exception:
        pass

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Erro: não consegui abrir a câmera.")

ear_window = deque(maxlen=SMOOTH_WINDOW)
ear_history = deque(maxlen=100)  # para gráfico
consec_closed = 0
closed_start_time = None
alert_played = False
phone_start_time = None

with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "ear", "pitch", "yaw", "is_phone", "eyes_open"])

# -------------------- Funções --------------------
def eye_aspect_ratio(eye_points):
    p1, p2, p3, p4, p5, p6 = eye_points
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    return (A + B) / (2.0 * (C + 1e-8))

PNP_IDX = [1, 152, 33, 263, 61, 291]
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0), (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)
], dtype=np.float32)

def estimate_head_pose(landmarks, img_w, img_h):
    image_points = np.array([landmarks[i] for i in PNP_IDX], dtype=np.float32)
    focal_length = img_w
    center = (img_w/2, img_h/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4,1))
    success, rotation_vec, _ = cv2.solvePnP(MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs)
    if not success:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rotation_vec)
    sy = math.sqrt(rmat[0,0]*rmat[0,0] + rmat[1,0]*rmat[1,0])
    if sy < 1e-6:
        x = math.atan2(-rmat[1,2], rmat[1,1])
        y = math.atan2(-rmat[2,0], sy)
        z = 0
    else:
        x = math.atan2(rmat[2,1], rmat[2,2])
        y = math.atan2(-rmat[2,0], sy)
        z = math.atan2(rmat[1,0], rmat[0,0])
    return math.degrees(y), math.degrees(x), math.degrees(z)

def detect_not_focused(landmarks, w, h, ear_smoothed):
    score = 0.0
    reasons = []
    yaw = pitch = 0.0
    try:
        yaw, pitch, roll = estimate_head_pose(landmarks, w, h)
        if abs(yaw) > 20:
            score += 0.4
            reasons.append(f"yaw={yaw:.1f}")
        if pitch > 15:
            score += 0.4
            reasons.append(f"pitch={pitch:.1f}")
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
            score += 0.15
        if dy > 0.08:
            score += 0.15
    except Exception:
        pass
    
    idle_sec = get_idle_seconds()
    if idle_sec > 5:
        score += 0.1
        reasons.append(f"idle={int(idle_sec)}s")
    
    if ear_smoothed is not None and pitch > 12 and ear_smoothed > 0.18:
        score += 0.2
        reasons.append("olhos_abertos_cabeca_baixa")
    
    score = min(1.0, score)
    return score >= 0.4, score, ";".join(reasons), yaw, pitch

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

# -------------------- HUD Drawing Functions --------------------
def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=10):
    """Desenha retângulo com cantos arredondados"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Linhas
    cv2.line(img, (x1+radius, y1), (x2-radius, y1), color, thickness)
    cv2.line(img, (x1+radius, y2), (x2-radius, y2), color, thickness)
    cv2.line(img, (x1, y1+radius), (x1, y2-radius), color, thickness)
    cv2.line(img, (x2, y1+radius), (x2, y2-radius), color, thickness)
    
    # Cantos
    cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness)

def draw_status_panel(frame, x, y, w, h, title, value, status_color, details=""):
    """Desenha painel de status moderno"""
    # Background semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), HUD_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Borda colorida
    draw_rounded_rect(frame, (x, y), (x+w, y+h), status_color, 2, radius=8)
    
    # Título
    cv2.putText(frame, title, (x+10, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Valor principal
    cv2.putText(frame, value, (x+10, y+55), cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2, cv2.LINE_AA)
    
    # Detalhes
    if details:
        cv2.putText(frame, details, (x+10, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

def draw_progress_bar(frame, x, y, w, h, progress, color, bg_color=(50, 50, 60)):
    """Barra de progresso animada"""
    # Background
    cv2.rectangle(frame, (x, y), (x+w, y+h), bg_color, -1)
    
    # Progresso
    fill_w = int(w * progress)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x+fill_w, y+h), color, -1)
    
    # Borda
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)

def draw_ear_graph(frame, x, y, w, h, ear_history):
    """Gráfico de EAR em tempo real"""
    if len(ear_history) < 2:
        return
    
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), HUD_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Linha threshold
    threshold_y = int(y + h - (EAR_THRESHOLD * h * 2))
    cv2.line(frame, (x, threshold_y), (x+w, threshold_y), HUD_ACCENT_WARNING, 1, cv2.LINE_AA)
    cv2.putText(frame, "Limite", (x+5, threshold_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, HUD_ACCENT_WARNING, 1, cv2.LINE_AA)
    
    # Desenha linha do gráfico
    points = []
    step = w / max(1, len(ear_history)-1)
    for i, ear_val in enumerate(ear_history):
        px = int(x + i * step)
        py = int(y + h - (ear_val * h * 2))  # escala
        py = max(y, min(y+h, py))
        points.append((px, py))
    
    if len(points) > 1:
        for i in range(len(points)-1):
            color = HUD_ACCENT_OK if ear_history[i] >= EAR_THRESHOLD else HUD_ACCENT_DANGER
            cv2.line(frame, points[i], points[i+1], color, 2, cv2.LINE_AA)
    
    # Borda
    draw_rounded_rect(frame, (x, y), (x+w, y+h), (100, 100, 120), 1, radius=5)

def draw_head_orientation(frame, x, y, size, yaw, pitch):
    """Indicador visual de orientação da cabeça"""
    center = (x + size//2, y + size//2)
    radius = size // 2 - 10
    
    # Círculo base
    cv2.circle(frame, center, radius, (60, 60, 80), 2)
    
    # Ponto indicando direção
    yaw_rad = math.radians(-yaw)  # inverte para intuitividade
    pitch_rad = math.radians(-pitch)
    
    offset_x = int(radius * 0.6 * math.sin(yaw_rad))
    offset_y = int(radius * 0.6 * math.sin(pitch_rad))
    
    point = (center[0] + offset_x, center[1] + offset_y)
    
    # Cor baseada em desvio
    deviation = math.sqrt(yaw**2 + pitch**2)
    if deviation < 15:
        color = HUD_ACCENT_OK
    elif deviation < 25:
        color = HUD_ACCENT_WARNING
    else:
        color = HUD_ACCENT_DANGER
    
    cv2.line(frame, center, point, color, 3, cv2.LINE_AA)
    cv2.circle(frame, point, 6, color, -1)
    
    # Labels
    cv2.putText(frame, f"Y:{yaw:.0f}", (x, y+size+15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.putText(frame, f"P:{pitch:.0f}", (x, y+size+30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

# -------------------- Loop Principal --------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ear_val = None
        eyes_open = False
        pitch = yaw = 0.0
        is_phone = False
        phone_score = 0.0

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            landmarks = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)

            # Calcula EAR
            left_ear = eye_aspect_ratio(landmarks[LEFT_EYE_IDX])
            right_ear = eye_aspect_ratio(landmarks[RIGHT_EYE_IDX])
            ear_val = (left_ear + right_ear) / 2.0
            
            ear_window.append(ear_val)
            ear_history.append(ear_val)
            ear_smoothed = float(np.mean(ear_window))

            # Head pose
            yaw, pitch, roll = estimate_head_pose(landmarks, w, h)

            # Detecta olhos fechados (real)
            is_ear_low = ear_smoothed < EAR_THRESHOLD
            is_head_down = pitch > PITCH_CLOSE_THRESHOLD
            is_head_turned = abs(yaw) > YAW_CLOSE_THRESHOLD
            is_real_eye_closed = is_ear_low and (not is_head_down) and (not is_head_turned)

            if is_real_eye_closed:
                consec_closed += 1
            else:
                consec_closed = 0

            eyes_open = not (consec_closed >= EYE_CLOSED_CONSEC_FRAMES)

            # Detecta phone/desfocado
            is_phone, phone_score, phone_reasons, _, _ = detect_not_focused(landmarks, w, h, ear_smoothed)

            # ========== NOVO HUD ==========
            
            # Painel: Status dos Olhos
            eye_status = "ABERTOS" if eyes_open else "FECHADOS"
            eye_color = HUD_ACCENT_OK if eyes_open else HUD_ACCENT_DANGER
            draw_status_panel(frame, 10, 10, 180, 80, "OLHOS", eye_status, eye_color, f"EAR: {ear_smoothed:.3f}")

            # Painel: Foco
            focus_status = "OK" if not is_phone else "DESFOCADO"
            focus_color = HUD_ACCENT_OK if not is_phone else HUD_ACCENT_WARNING
            draw_status_panel(frame, 200, 10, 180, 80, "FOCO", focus_status, focus_color, f"Score: {phone_score:.2f}")

            # Painel: Orientação da cabeça
            draw_head_orientation(frame, 390, 10, 80, yaw, pitch)

            # Gráfico de EAR
            draw_ear_graph(frame, 10, h-130, 300, 120, ear_history)

            # Alerta de inatividade (barra de progresso)
            now = time.time()
            if not eyes_open:
                if closed_start_time is None:
                    closed_start_time = now
                    alert_played = False
                elapsed = now - closed_start_time
                progress = min(1.0, elapsed / ALERT_SECONDS)
                
                # Barra de alerta
                bar_w = 250
                bar_x = w - bar_w - 20
                bar_y = 20
                draw_progress_bar(frame, bar_x, bar_y, bar_w, 30, progress, HUD_ACCENT_DANGER)
                cv2.putText(frame, f"INATIVO: {int(elapsed)}s / {ALERT_SECONDS}s", 
                           (bar_x+5, bar_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                if elapsed >= ALERT_SECONDS and not alert_played:
                    if MINIMIZE_WINDOWS_ON_ALERT:
                        try:
                            my_hwnd = GetForegroundWindow()
                            minimize_all_other_windows(except_hwnd=my_hwnd)
                        except Exception:
                            pass

                    play_alert_audio_once()

                    alert_win = "ALERTA - FOQUE!"
                    cv2.namedWindow(alert_win, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(alert_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                    flash_on = True
                    flash_timer = time.time()

                    while True:
                        ret2, f2 = cap.read()
                        if not ret2:
                            break
                        rgb2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
                        res2 = face_mesh.process(rgb2)
                        eyes_open_now = False
                        if res2.multi_face_landmarks:
                            lm2 = res2.multi_face_landmarks[0]
                            landmarks2 = np.array([[p.x * w, p.y * h] for p in lm2.landmark], dtype=np.float32)
                            le = eye_aspect_ratio(landmarks2[LEFT_EYE_IDX])
                            re = eye_aspect_ratio(landmarks2[RIGHT_EYE_IDX])
                            if (le + re) / 2.0 >= EAR_THRESHOLD:
                                eyes_open_now = True

                        disp = f2.copy()
                        h2, w2 = disp.shape[:2]
                        overlay = disp.copy()
                        alpha = 0.6 if flash_on else 0.25
                        cv2.rectangle(overlay, (0,0), (w2, h2), (0,0,255), -1)
                        cv2.addWeighted(overlay, alpha, disp, 1 - alpha, 0, disp)

                        cv2.putText(disp, "ACORDA!", (int(w2*0.05), int(h2*0.45)),
                                    cv2.FONT_HERSHEY_DUPLEX, 2.0, (255,255,255), 4, cv2.LINE_AA)

                        small = cv2.resize(f2, (int(w2*0.25), int(h2*0.25)))
                        disp[10:10+small.shape[0], 10:10+small.shape[1]] = small

                        cv2.imshow(alert_win, disp)

                        if time.time() - flash_timer > 0.5:
                            flash_on = not flash_on
                            flash_timer = time.time()

                        if eyes_open_now:
                            try:
                                if PYGAME_AVAILABLE and pygame.mixer.get_init():
                                    pygame.mixer.music.stop()
                            except Exception:
                                pass
                            break

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    cv2.destroyWindow(alert_win)
                    closed_start_time = None
                    alert_played = False
            else:
                closed_start_time = None
                alert_played = False

            # Phone alert timing
            if is_phone:
                if phone_start_time is None:
                    phone_start_time = now
                phone_elapsed = now - phone_start_time
                if phone_elapsed >= PHONE_ALERT_SECONDS:
                    play_alert_audio_once()
                else:
                    # Barra de phone warning
                    phone_progress = phone_elapsed / PHONE_ALERT_SECONDS
                    bar_w = 200
                    bar_x = w - bar_w - 20
                    bar_y = 60
                    draw_progress_bar(frame, bar_x, bar_y, bar_w, 20, phone_progress, HUD_ACCENT_WARNING)
                    cv2.putText(frame, f"Celular: {int(phone_elapsed)}s", 
                               (bar_x+5, bar_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                phone_start_time = None

        else:
            # Sem rosto detectado
            cv2.putText(frame, "ROSTO NAO DETECTADO", (w//2-150, h//2), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, HUD_ACCENT_DANGER, 2, cv2.LINE_AA)
            ear_window.clear()
            consec_closed = 0
            closed_start_time = None
            alert_played = False

        cv2.imshow("Face+Eyes Detector", frame)

        # Logging
        if ear_val is not None:
            try:
                with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.utcnow().isoformat(), f"{ear_val:.4f}", 
                                   f"{pitch:.2f}", f"{yaw:.2f}", int(is_phone), int(eyes_open)])
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
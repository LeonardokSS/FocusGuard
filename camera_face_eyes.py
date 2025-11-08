"""
camera_face_eyes.py
Protótipo N1 -> detecta rosto e calcula EAR (abertura dos olhos).
Requisitos: mediapipe, opencv-python, numpy, pygame (opcional para MP3)
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

# -------------------- Windows helper (topmost + minimizar outras janelas) --------------------
user32 = ctypes.WinDLL('user32', use_last_error=True)

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

EnumWindows = user32.EnumWindows
# define tipo de callback
WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

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
                return True  # pular
            length = GetWindowTextLength(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            GetWindowText(hwnd, buff, length + 1)
            title = buff.value or ""
            # pular janelas sem título e a exceção
            if not title:
                return True
            if except_hwnd and hwnd == except_hwnd:
                return True
            if exclude_title_contains and exclude_title_contains.lower() in title.lower():
                return True
            # minimizar
            ShowWindow(hwnd, SW_MINIMIZE)
        except Exception:
            pass
        return True
    enum_proc = wintypes.WNDENUMPROC(callback)
    EnumWindows(enum_proc, 0)

# -------------------- Configurações do detector --------------------
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.20       # abaixo disso consideramos olho "fechado"
CONSEC_FRAMES = 3         # frames consecutivos para confirmar "fechado"
SMOOTH_WINDOW = 5         # média móvel de EAR em N frames
LOG_CSV = "eye_log.csv"   # arquivo de log (timestamp, ear, eyes_open)

# alerta / comportamento
ALERT_SECONDS = 15        # segundos para disparar o alerta (ajuste aqui)
MINIMIZE_WINDOWS_ON_ALERT = True  # True = minimiza outras janelas quando alerta disparar
ALERT_AUDIO_FILE = "alert.mp3"    # arquivo MP3 (coloque na mesma pasta do script)

# -------------------- Inicializações --------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# prepara pygame para tocar mp3 (se disponível)
try:
    import pygame
    pygame.mixer.pre_init(44100, -16, 2, 2048)
    pygame.init()
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

# inicializa câmera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Erro: não consegui abrir a câmera. Verifique índice ou drivers.")

# smoothing e contadores
ear_window = deque(maxlen=SMOOTH_WINDOW)
consec_closed = 0

# controle de alerta (deve ficar FORA do loop)
closed_start_time = None
alert_played = False

# cria CSV de log (cabeçalho)
with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "ear", "eyes_open"])

# função EAR
def eye_aspect_ratio(eye_points):
    p1, p2, p3, p4, p5, p6 = eye_points
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    ear = (A + B) / (2.0 * (C + 1e-8))
    return ear

# helper para tocar áudio (reutilizável)
def play_alert_audio_once():
    global alert_played
    if alert_played:
        return
    try:
        if PYGAME_AVAILABLE:
            # tenta carregar e tocar (não bloqueia)
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

            if ear_smoothed < EAR_THRESHOLD:
                consec_closed += 1
            else:
                consec_closed = 0

            eyes_open = not (consec_closed >= CONSEC_FRAMES)

            # desenho: pontos dos olhos
            for (x, y) in np.vstack([left_eye_pts, right_eye_pts]).astype(int):
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            status_text = f"EAR: {ear_smoothed:.2f}  Open: {eyes_open}"
            color = (0, 255, 0) if eyes_open else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            x_coords = [int(p[0]) for p in landmarks]
            y_coords = [int(p[1]) for p in landmarks]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

            # ---------- controle de tempo real para alerta ----------
            now = time.time()
            if not eyes_open:
                if closed_start_time is None:
                    closed_start_time = now
                    alert_played = False

                elapsed = now - closed_start_time
                cv2.putText(frame, f"Inativo: {int(elapsed)}s", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if elapsed >= ALERT_SECONDS and not alert_played:
                    # minimiza janelas se configurado
                    if MINIMIZE_WINDOWS_ON_ALERT:
                        try:
                            my_hwnd = GetForegroundWindow()
                            minimize_all_other_windows(except_hwnd=my_hwnd, exclude_title_contains=None)
                        except Exception:
                            pass

                    # toca o áudio (uma vez) e abre a janela de alerta
                    play_alert_audio_once()

                    # cria janela de alerta em fullscreen e força topmost
                    alert_win = "ALERTA - ACORDA!"
                    cv2.namedWindow(alert_win, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(alert_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    # tentar setar topmost algumas vezes
                    for _ in range(8):
                        set_window_topmost_by_name(alert_win, top=True)
                        time.sleep(0.03)

                    # loop do estado de alerta (piscar) — continua até eyes_open True
                    flash_on = True
                    flash_timer = time.time()
                    while True:
                        # atualiza frame para mostrar vídeo em tempo real dentro da janela de alerta
                        ret2, f2 = cap.read()
                        if not ret2:
                            break
                        # recalcula landmarks rápido
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
                            # simples threshold aqui
                            if ear_now >= EAR_THRESHOLD:
                                eyes_open_now = True

                        # cria display com overlay piscante
                        disp = f2.copy()
                        h2, w2 = disp.shape[:2]
                        overlay = disp.copy()
                        alpha = 0.6 if flash_on else 0.25
                        cv2.rectangle(overlay, (0,0), (w2, h2), (0,0,255), -1)
                        cv2.addWeighted(overlay, alpha, disp, 1 - alpha, 0, disp)
                        cv2.putText(disp, "ACORDA! ABRA OS OLHOS", (int(w2*0.05), int(h2*0.5)),
                                    cv2.FONT_HERSHEY_DUPLEX, 2.0, (255,255,255), 4, cv2.LINE_AA)
                        # mostra a câmera num canto
                        small = cv2.resize(f2, (int(w2*0.25), int(h2*0.25)))
                        disp[10:10+small.shape[0], 10:10+small.shape[1]] = small

                        cv2.imshow(alert_win, disp)

                        # piscar a cada 0.5s
                        if time.time() - flash_timer > 0.5:
                            flash_on = not flash_on
                            flash_timer = time.time()

                        # sair do alerta se abriu os olhos
                        if eyes_open_now:
                            break
                        # permitir cancelar com 'q'
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    # fim do loop de alerta: fecha janela de alerta e limpa variáveis
                    try:
                        cv2.destroyWindow(alert_win)
                        set_window_topmost_by_name(alert_win, top=False)
                    except Exception:
                        pass
                    closed_start_time = None
                    alert_played = False
            else:
                # usuário reabriu os olhos — reset
                closed_start_time = None
                alert_played = False

        else:
            # sem rosto detectado: reseta janela e flags
            ear_window.clear()
            consec_closed = 0
            closed_start_time = None
            alert_played = False
            cv2.putText(frame, "Rosto nao detectado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # mostra frame principal (pequena janela)
        cv2.imshow("Face+Eyes Detector", frame)

        # log (apenas se ear_val calculado)
        if ear_val is not None:
            with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.utcnow().isoformat(), f"{ear_val:.4f}", int(eyes_open)])

        # tecla para sair
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

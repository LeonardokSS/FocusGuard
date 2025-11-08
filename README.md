# FocusGuard
FocusGuard é um projeto em Python que usa visão computacional para detectar quando você está dormindo ou perdendo o foco no PC. Ele usa a webcam para medir a abertura dos olhos. Se você ficar tempo demais de olhos fechados, o sistema dispara um alerta: toca um áudio MP3, mostra uma mensagem piscando e força a janela principal a ficar em primeiro plano.

Como usar

clone o projeto:

git clone https://github.com/SEU-USUARIO/FocusGuard.git
cd FocusGuard


crie o ambiente virtual:

python -m venv .venv
.venv\Scripts\activate


instale dependências:

pip install -r requirements.txt


coloque o arquivo alert.mp3 dentro da pasta do projeto

rode:

python camera_face_eyes.py


Esse projeto foi feito em Python 3.12 usando OpenCV, MediaPipe e funções do Windows (ctypes).
Por enquanto funciona apenas no Windows.

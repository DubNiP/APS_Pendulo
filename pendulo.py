import cv2
import numpy
import csv
import time
import pandas
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


vdo = 'pendulo.mp4'
cap = cv2.VideoCapture(vdo)
fps = cap.get(cv2.CAP_PROP_FPS)
dadosVdo = 'dados_pendulo.csv'

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

_q=110
_w=30
_e=0
_a=200
_s=200
_d=200
#110,30,0,200,200,200

lim_cor_min = numpy.array([100, 30, 0])      
lim_cor_max = numpy.array([180, 255, 255])  
kernel_opening = numpy.ones((5, 5), numpy.uint8)
kernel_closing = numpy.ones((5, 5), numpy.uint8)

with open(dadosVdo, 'w', newline='') as arquivo_csv:
    escritor_csv = csv.writer(arquivo_csv)
    escritor_csv.writerow(['Frame', 'Tempo (s)', 'FPS', 'posição X', 'posição Y'])
    frame_num = 0

    inicio = int(time.time() * 1000)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        

        key = cv2.waitKey(1)
        #print(f"fps: {fps}")
        
        if key & 0xFF == ord('Q'):
            _q = _q + 10;
        if key & 0xFF == ord('W'):
            _w = _w + 10;
        if key & 0xFF == ord('E'):
            _e = _e + 10;
        if key & 0xFF == ord('A'):
            _a = _a + 10;
        if key & 0xFF == ord('S'):
            _s = _s + 10;
        if key & 0xFF == ord('D'):
            _d = _d + 10;
        
        if key & 0xFF == ord('q'):
            _q = _q - 10;
        if key & 0xFF == ord('w'):
            _w = _w - 10;
        if key & 0xFF == ord('e'):
            _e = _e - 10;
        if key & 0xFF == ord('a'):
            _a = _a - 10;
        if key & 0xFF == ord('s'):
            _s = _s - 10;
        if key & 0xFF == ord('d'):
            _d = _d - 10;

        print(f'Limites: {_q}, {_w}, {_e}, {_a}, {_s}, {_d}')

        lim_cor_min = numpy.array([_q, _w, _e])      
        lim_cor_max = numpy.array([_a, _s, _d])
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, lim_cor_min, lim_cor_max)
        mascara2 = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel_opening)
        mascara3 = cv2.morphologyEx(mascara2, cv2.MORPH_CLOSE, kernel_closing)
        contornos = cv2.findContours(mascara3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contornos) > 0:

            # Percorre todos os contornos
            Mcentro = None
            Mraio = None
            for contorno in contornos[0]:
                # Obtém o círculo mínimo que envolve o contorno
                (x, y), raio = cv2.minEnclosingCircle(contorno)
                centro = (int(x), int(y))
                raio = int(raio)
                if( Mraio is None or raio > Mraio):
                    Mcentro = centro
                    Mraio = raio

                # Desenha o círculo na imagem original
            if Mcentro is not None and Mraio is not None:
                tempo = frame_num / fps
                frame = cv2.circle(frame, Mcentro, Mraio, (0, 255, 0), 2)  # cor verde e espessura 2
                escritor_csv.writerow([frame_num, tempo, fps, Mcentro[0], Mcentro[1]])

        frame_num += 1
        agora = int(time.time() * 1000)
        espera = (inicio + ((1000*frame_num/fps))) - agora
        if espera > 0:
            #print(f"Esperando {espera} ms para o próximo frame.")
            time.sleep(espera / 1000.0)  # Converte milissegundos para segundos
        else:
            inicio = agora - (1000*frame_num/fps) 

        cv2.imshow('Pendulo', frame)
        #cv2.imshow('Mascara3', mascara3)
        if key & 0xFF == ord('n'):
            break
    cap.release()
    cv2.destroyAllWindows()

dados = pandas.read_csv(dadosVdo, encoding='latin1')

def MHA(t,A,b,omega,phi):
    return A * numpy.exp((-b * t)/(2*m)) * numpy.cos(omega * t + phi)

if len(dados) > 0:
    t = dados['Tempo (s)'].values
    x = dados['posição X'].values
    x = x - numpy.mean(x)
    m = 0.252
    A = 174
    b = 0.001
    omega = 2.62
    phi = 0

    try:
        popt, pcov = curve_fit(MHA, t, x, p0=[A, b, omega, phi], maxfev=5000)
        A, b, omega, phi = popt

        T= 2 * numpy.pi / omega
        Q = 2 * numpy.pi / (1-numpy.exp(-2*b * T))
        print(f"Fator de qualidade Q: {Q}")

        print(f"Amplitude ajustada: {A}")
        print(f"Coeficiente de amortecimento ajustado: {b}")
        print(f"Frequência angular ajustada: {omega}")
        print(f"Fase ajustada: {phi}")

        plt.figure(figsize=(14, 7))
        plt.plot(t, x, 'o', label='Dados originais')
        plt.plot(t, MHA(t, *popt), 'r-', label='Ajuste da curva')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Posição X')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    except RuntimeError as e:
        print(f"Erro ao ajustar a curva: {e}")
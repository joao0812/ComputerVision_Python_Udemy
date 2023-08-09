import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture('24 - Motion Tracking with Mean Shift and CAMSHIFT\\assets\\calculadoraVideo.mp4')
    #cap = cv2.VideoCapture(0)
    x1, y1 = 100, 500
    w1, h1 = 350, 550

    first_ret, first_frame = cap.read()
    #first_frame = first_frame[::3, ::3]
    print(first_frame.shape)

    calc = first_frame[y1:y1+h1, x1:x1+w1]
    first_frame_hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
    hsv_calc = cv2.cvtColor(calc, cv2.COLOR_BGR2HSV)

    hist_calc = cv2.calcHist([hsv_calc], [0], None, [180], [0, 180])
    # A função cv2.calcHist() é utilizada para calcular o histograma de uma ou mais imagens em relação aos valores de seus pixels. O histograma é uma representação gráfica da distribuição das intensidades de cores em uma imagem
    # cv2.calcHist(images, channels, mask, histSize, ranges)
    # images: É a lista de imagens de entrada sobre as quais o histograma será calculado. Essas imagens devem ser fornecidas como uma lista (ou tupla) de arrays NumPy.
    # channels: É a lista de índices dos canais das imagens para os quais o histograma será calculado. Por exemplo, para uma imagem BGR, o canal 0 representa o canal azul (Blue), o canal 1 representa o canal verde (Green) e o canal 2 representa o canal vermelho (Red). Portanto, [0] calculará o histograma do canal azul, [1] do canal verde e [2] do canal vermelho.
    # mask: É uma máscara binária opcional. Se fornecida, o histograma será calculado apenas para os pixels onde a máscara for diferente de zero. Isso é útil para calcular o histograma de uma região de interesse específica.
    # histSize: É o número de bins (intervalos) usados para calcular o histograma. Por exemplo, [256] significa que o histograma terá 256 bins.
    # ranges: São os intervalos dos valores dos pixels que serão considerados para calcular o histograma. Por exemplo, [0, 256] indica que o histograma será calculado para valores de pixels no intervalo de 0 a 255.

    # Normalizando os dados do histograma
    hist_calc = cv2.normalize(hist_calc, hist_calc, 0, 255, cv2.NORM_MINMAX)

    print(hist_calc)

    print(first_frame.shape)

    while True:
        ret, frame = cap.read()
        #frame = frame[::3, ::3]

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_calc = cv2.calcBackProject(
            [hsv_frame], [0], hist_calc, [0, 180], 1)

        term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        _, tranck_window = cv2.meanShift(mask_calc, (x1,y1, w1, h1), term_criteria)
        tranck_window_X, tranck_window_Y = tranck_window[0], tranck_window[1]
        tranck_window_W, tranck_window_H = tranck_window[2], tranck_window[3]
        print(tranck_window)
        print(tranck_window_X)
        print(tranck_window_Y)
        print(tranck_window_W)
        print(tranck_window_H)


        if ret and first_ret:
            cv2.rectangle(frame, (tranck_window_X, tranck_window_Y), (tranck_window_X+tranck_window_W, tranck_window_Y+tranck_window_H), (255,0,0), 2)
            cv2.imshow('Crop', calc)
            cv2.imshow('hsv', hsv_calc)
            cv2.imshow('Mask', mask_calc)
            cv2.imshow('Video', frame)
            cv2.imshow('Video HSV', first_frame_hsv)

            key = cv2.waitKey(30)
            if key == 27:  # ESC
                break
        else:
            print('ERRO')
            break

    cap.release()
    cv2.destroyAllWindows()


if '__main__' == __name__:
    main()

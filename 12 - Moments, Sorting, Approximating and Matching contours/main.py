import os
import sys

import cv2

import pandas as pd
import numpy as np

import matplotlib as mlp


def moments(formasGeometricas, formasGeometricas_gray):
    # A função cv2.Canny() é um método da biblioteca OpenCV que realiza a detecção de bordas em uma imagem usando o algoritmo de Canny. O método Canny é um dos métodos mais populares e eficazes para a detecção de bordas em imagens.
    edged = cv2.Canny(formasGeometricas_gray, 50, 200)
    # image: É a imagem de entrada na qual a detecção de bordas será aplicada. Geralmente, é uma imagem em escala de cinza.
    # threshold1: É o primeiro valor de limiar para a histerese da detecção de bordas. Geralmente, é um valor baixo.
    # threshold2: É o segundo valor de limiar para a histerese da detecção de bordas. Geralmente, é um valor alto.
    # apertureSize (opcional): É o tamanho do operador de Sobel usado para calcular os gradientes. O valor padrão é 3.
    # L2gradient (opcional): Indica se é usado o cálculo de gradiente L2-norm. O valor padrão é False.

    cv2.imshow('Formas Geometricas', formasGeometricas_gray)
    cv2.imshow('Edges', edged)

    contours, hierarchy = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # A função cv2.findContours() é um método da biblioteca OpenCV que encontra os contornos de objetos em uma imagem binária ou em uma imagem com valores de intensidade representando as bordas.
    # A função cv2.findContours() retorna os contornos encontrados e a hierarquia dos contornos (estrutura que descreve as relações entre os contornos, como contorno pai e contorno filho). Os contornos são retornados como uma lista de arrays numpy, onde cada array representa um contorno, contendo coordenadas dos pontos do contorno.
    # image: É a imagem de entrada na qual os contornos serão encontrados. Geralmente, é uma imagem binária ou uma imagem com bordas detectadas.
    # mode: É o modo de busca de contornos. Pode ser cv2.RETR_EXTERNAL (busca apenas pelos contornos externos), cv2.RETR_LIST (busca todos os contornos sem hierarquia) e outros modos disponíveis.
    # method: É o método de aproximação de contornos. Pode ser cv2.CHAIN_APPROX_SIMPLE (aproximação simples), cv2.CHAIN_APPROX_NONE (sem aproximação) e outros métodos disponíveis.
    # contours (opcional): É a lista de contornos encontrados. Se não for fornecida, uma nova lista será criada.
    # hierarchy (opcional): É a representação hierárquica dos contornos. Se não for fornecida, uma nova lista será criada.
    # offset (opcional): É um valor opcional a ser adicionado a todos os pontos dos contornos.

    print(f'Number of contours found is: {len(contours)}')
    cv2.drawContours(formasGeometricas, contours, -1, (0, 0, 0), 3)

    cv2.imshow('Formas Geometricas', formasGeometricas)

    for c in contours:
        area = cv2.contourArea(c)
        print(area)

    # print('--'*30)
    sorted_areas = sorted(contours, key=cv2.contourArea, reverse=True)
    """ for c_sort in sorted_areas:
        print(cv2.contourArea(c_sort)) """

    print('--'*30)
    for (i, c_sort2) in enumerate(sorted_areas):
        # A função cv2.moments() em OpenCV é usada para calcular os momentos de uma imagem ou de um contorno específico. Esses momentos são estatísticas que descrevem diferentes características da distribuição de intensidade ou forma na imagem.
        # A função cv2.moments() retorna um dicionário de momentos calculados com base na entrada fornecida. Esses momentos incluem momentos espaciais (como a área) e momentos centrais (como o centroide)
        M = cv2.moments(c_sort2)
        # m00: Momento de ordem zero, que representa a área ou soma dos pixels da região.
        # m10, m01: Momentos de primeira ordem, que são usados para calcular o centroide (coordenadas x e y) da região.
        # mu20, mu11, mu02: Momentos centrais de segunda ordem, que descrevem a distribuição dos pixels ao redor do centroide.
        # nu20, nu11, nu02: Momentos centrais normalizados de segunda ordem, que são calculados a partir dos momentos centrais de segunda ordem e são invariantes a rotações, escalas e translações.
        min_point_x = min(c_sort2, key=lambda pnt: pnt[0][0])
        min_point_y = min(c_sort2, key=lambda pnt: pnt[0][1])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        # Esses momentos podem ser usados para realizar várias tarefas, como cálculos de propriedades geométricas, detecção de características, classificação de objetos, entre outros.

        print(f'{i} - Area: {cv2.contourArea(c_sort2)}\n Centroid: ({cx},{cy})\n Text Point: ({min_point_x}, {min_point_y})')
        cv2.circle(formasGeometricas, (cx, cy), 10, (0, 0, 0), -1)
        cv2.putText(formasGeometricas, f'{i}', (
            min_point_x[0][0], min_point_y[0][1]), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)
    # print(contours)
    # print(hierarchy)

    cv2.imshow('Formas Geometricas', formasGeometricas)
    cv2.waitKey(0)


def polyAprox(casaDesenho, casaDesenho_gray, casaDesenho_copy):
    # A função cv2.threshold() em OpenCV é utilizada para realizar a limiarização (thresholding) de uma imagem. A limiarização é uma técnica de segmentação de imagens que permite converter uma imagem em escala de cinza ou em tons de cinza para uma imagem binária, onde os pixels são classificados em dois grupos: valores acima de um determinado limite são atribuídos a um valor específico (por exemplo, 255 ou branco), enquanto os valores abaixo do limite são atribuídos a outro valor (por exemplo, 0 ou preto).
    ret, thresh = cv2.threshold(
        casaDesenho_gray, 190, 255, cv2.THRESH_BINARY_INV)
    # src: É a imagem de origem, que deve ser uma imagem em escala de cinza ou em tons de cinza.
    # thresh: É o valor do limite para a limiarização.
    # maxval: É o valor atribuído aos pixels que estão acima do limite.
    # type: É o tipo de limiarização a ser aplicado. Pode ser um dos seguintes valores:
    # cv2.THRESH_BINARY: Os pixels acima do limite são definidos como maxval, enquanto os pixels abaixo são definidos como 0.
    # cv2.THRESH_BINARY_INV: Os pixels acima do limite são definidos como 0, enquanto os pixels abaixo são definidos como maxval.
    # cv2.THRESH_TRUNC: Os pixels acima do limite são definidos como o próprio limite, enquanto os pixels abaixo permanecem inalterados.
    # cv2.THRESH_TOZERO: Os pixels acima do limite permanecem inalterados, enquanto os pixels abaixo são definidos como 0.
    # cv2.THRESH_TOZERO_INV: Os pixels acima do limite são definidos como 0, enquanto os pixels abaixo permanecem inalterados.
    # dst (opcional): É a imagem de destino, onde o resultado da limiarização será armazenado.
    # ret: É o valor de limiar usado na limiarização.
    # thresh: É a imagem binária resultante após a aplicação da limiarização.
    cv2.imshow('Binary', thresh)

    casaCountours, casaHierarchy = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # BOUNDING REACT
    for conto in casaCountours:
        # A função cv2.boundingRect() em OpenCV é utilizada para calcular o retângulo delimitador de um contorno. Esse retângulo é o menor retângulo vertical alinhado aos eixos que envolve completamente o contorno.
        x, y, w, h = cv2.boundingRect(conto)
        # A função cv2.boundingRect() retorna as coordenadas do retângulo delimitador: a coordenada x do canto superior esquerdo (x), a coordenada y do canto superior esquerdo (y), a largura do retângulo (w) e a altura do retângulo (h).
        cv2.rectangle(casaDesenho, (x, y), (x+w, y+h), (0, 0, 255), 2)

    for conto2 in casaCountours:
        # calcula a precisão da aproximação poligonal com base no comprimento do contorno original conto2. A função cv2.arcLength() retorna o comprimento do contorno. Nesse caso, a precisão é definida como 3% do comprimento do contorno.
        accuracy = 0.03 * cv2.arcLength(conto2, True)
        # A função cv2.approxPolyDP() é usada para realizar a aproximação poligonal do contorno original conto2 com a precisão calculada. O resultado é armazenado na variável approx.
        approx = cv2.approxPolyDP(conto2, accuracy, True)
        cv2.drawContours(casaDesenho_copy, [approx], 0, (255, 0, 0), 2)
    cv2.imshow('Casa', casaDesenho)
    cv2.imshow('Casa Copy', casaDesenho_copy)
    cv2.waitKey(0)


def convexHull(trapezio, formasGeometricas):
    trapezio_gray = cv2.cvtColor(trapezio, cv2.COLOR_RGB2GRAY)
    formasGeometricas_gray = cv2.cvtColor(
        formasGeometricas, cv2.COLOR_RGB2GRAY)

    cv2.imshow('trapezio', trapezio_gray)
    cv2.imshow('formas', formasGeometricas_gray)

    ret, thresh1 = cv2.threshold(trapezio_gray, 210, 255, 0)
    ret, thresh2 = cv2.threshold(formasGeometricas_gray, 210, 255, 0)

    contours_trapezio, hierarchy = cv2.findContours(
        thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours_formas, hierarchy_gray = cv2.findContours(
        thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow('thresh1', thresh1)
    cv2.imshow('thresh2', thresh2)

    sorted_contours_trapezio = sorted(
        contours_trapezio[1], key=cv2.contourArea, reverse=True)
    sorted_contours_formas = sorted(
        contours_formas, key=cv2.contourArea, reverse=True)

    for (i, c_sort2) in enumerate(sorted_contours_formas):

        M = cv2.moments(c_sort2)

        min_point_x = min(c_sort2, key=lambda pnt: pnt[0][0])
        min_point_y = min(c_sort2, key=lambda pnt: pnt[0][1])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        print(f'{i} - Area: {cv2.contourArea(c_sort2)}\n Centroid: ({cx},{cy})\n Text Point: ({min_point_x}, {min_point_y})')
        cv2.circle(formasGeometricas, (cx, cy), 10, (255, 0, 0), -1)
        cv2.putText(formasGeometricas, f'{i}', (
            min_point_x[0][0], min_point_y[0][1]), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)
    # print(contours)
    # print(hierarchy)

    cv2.drawContours(trapezio, contours_trapezio, -1, (0, 0, 255), 3)
    cv2.drawContours(formasGeometricas, contours_formas, -1, (0, 0, 255), 3)

    cv2.imshow('trapezio', trapezio)
    cv2.imshow('formas', formasGeometricas)

    for (i, c) in enumerate(contours_formas):
        match = cv2.matchShapes(contours_trapezio[1], c, 3, 0.0)
        print(match)
        if match < 0.009:
            closest_contour = c
            cv2.drawContours(formasGeometricas, [closest_contour], -1, (0, 0, 0), 3)
        else:
            closest_contour = []

    
    cv2.imshow('RESULT', formasGeometricas)
    cv2.waitKey(0)
    pass


def main():
    formasGeometricas = cv2.imread('./assets/formasGeometricas.jpg')
    casaDesenho = cv2.imread('./assets/casaDesenho.PNG')
    trapezio = cv2.imread('./assets/trapezio.jpg')
    circulo = cv2.imread('./assets/circulo.jpg')
    hexagono = cv2.imread('./assets/hexagono.jpg')

    casaDesenho = cv2.resize(
        casaDesenho, (casaDesenho.shape[1]*2, casaDesenho.shape[0]*2))
    casaDesenho_copy = casaDesenho.copy()

    formasGeometricas_gray = cv2.cvtColor(
        formasGeometricas, cv2.COLOR_BGR2GRAY)
    casaDesenho_gray = cv2.cvtColor(casaDesenho, cv2.COLOR_BGR2GRAY)

    #moments(formasGeometricas, formasGeometricas_gray)

    #polyAprox(casaDesenho, casaDesenho_gray, casaDesenho_copy)

    convexHull(hexagono, formasGeometricas)


if '__main__' == __name__:
    main()

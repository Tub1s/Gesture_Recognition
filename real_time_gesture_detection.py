import cv2
import numpy as np
import math
import keyboard

# Okreslenie, z ktorej kamery chcemy korzystac. Jesli uzytkownik posiada wiecej niz 1 wtedy parametr w VideCapture ustawiamy na 1 lub 2
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    # Odczytujemy obraz z kamery
    ret, img = cap.read()

    # Inicjujejmy kwadrat
    cv2.rectangle(img, (300,300), (100,100), (0,255,0),0)
    crop_img = img[100:300, 100:300]

    # Konwertujemy na odcienie szarosci
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # Korzystamy z rozmycia gausowskiego 
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)

    # Korzystamy z metody Otsu sluzacej do binaryzacji czyli progowania obrazu. Wykonujemy konwersje obrazu w odcieniach szarosci do obrazu binarnego.
    _, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Wyswietlamy obraz po zastosowaniu metody Otsu
    cv2.imshow('Thresholded', thresh1)

    # Sprawdzenie wersji OpenCV, aktualna jest wersja 4.4 (Robimy to ze wzgledu na parametry, kiedys trzeba bylo podac 3)
    (version, _, _) = cv2.__version__.split('.')

    #Uzyskanie zarysu/konturu reki
    if version == '4':
        contours, hierarchy = cv2.findContours(thresh1.copy(), \
               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
   
    # Szukamy maksymalny kontur wypelniajacy obszar
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    # Tworzymy ROI (Region of Interest) w okol naszej reki, mozemy to zrobic tylko po znalezieniu konturu
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    # Poszukujemy Convex Hull - otoczki wypuklej
    hull = cv2.convexHull(cnt)

    # Rysujemy kontury
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    # Szukamy otoczki wypuklej
    hull = cv2.convexHull(cnt, returnPoints=False)

    # Poszukujemy defektow wypuklosci (convexity defects)
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    # Korzystamy z reguly cosinusa w poszukiwaniu kata pomiedzy palcami
    # Jesli kat jest wiekszy niz 90 ignorujemy "defekty"
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        # obliczanie dlugosci trojkata
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # Implementacja reguly cosinusa w poszukiwaniu kata pomiedzy palcami
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # Ignorowanie kata wiekszego niz 90, reszte defektow oznaczamy czerwona kropka pomiedzy palcami
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 1, [0,0,255], -1)
        #dist = cv2.pointPolygonTest(cnt,far,True)

        # Narysowanie linni dla punktow wypuklosciw
        cv2.line(crop_img,start, end, [0,255,0], 2)

    # Wyswietlanie rozpoznanego gestu/liczby palcow
    if count_defects == 1:
        cv2.putText(img,"Peace", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        #keyboard.press('w')
        keyboard.press_and_release('k')  
    elif count_defects == 2:
        str = "Katniss Everdeen Salute"
        cv2.putText(img, str, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        #keyboard.press('a')
        keyboard.press_and_release('c')
    elif count_defects == 3:
        cv2.putText(img,"Yankee Salute", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        #keyboard.press('s')
        keyboard.press_and_release('left')
    elif count_defects == 4:
        cv2.putText(img,"Hello", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        #keyboard.press('d')
        keyboard.press_and_release('right')

    #
    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)

    k = cv2.waitKey(10)
    if k == 27:
        break

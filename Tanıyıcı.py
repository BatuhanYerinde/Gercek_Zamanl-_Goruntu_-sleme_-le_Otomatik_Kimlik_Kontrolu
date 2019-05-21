import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()  # tanıyıcı nesne olusturuluyor.
recognizer.read('trainer/trainer.yml')  # egitim dosyası okunuyor
cascadePath = "haarcascade_frontalface_default.xml"  # haarcascade yuz tanıma yolu belirtiliyor.
faceCascade = cv2.CascadeClassifier(cascadePath)  # yuz tanıma nesnesi olusturuluyor.

font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0

# veri setleri icimde bulunan isimler. Listeye kaydediliyor.
names = ['None', 'Batu', 'Muho', 'Selim']

# video cekmek icin nesne olusturuluyor.
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # video genislik bilgisi.
cam.set(4, 480)  # video yukseklik bilgisi.

#  yuzu algılamak icin minimum pencere boyutları nesneye kaydediliyor.
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:

    ret, img = cam.read()  # resimler okunuyor.
    img = cv2.flip(img, -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # gray levela cevriliyor.

    faces = faceCascade.detectMultiScale(
        # cascade modülü icerisindeki detectmultiScale fonksiyonunun parametreleri giriliyor.
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:  # faces listesi icerisinde dolanılarak dikdortgen ciziliyor.

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(
            gray[y:y + h, x:x + w])  # tahmin islemleri gerceklestirilip id ve confidence degerleri kaydediliyor.

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (
                confidence < 100):  # güven degeri 100 den kucukse id degerleri names listesinin icerisine koyuluyor ve listedeki isimlere erisiliyor.
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)  # isimler cerceve uzerine yazdırılıyor.
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0),
                    1)  # guven dogruluk degeri yazdırılıyor.

    cv2.imshow('camera', img)  # resim gosteriliyor.

    k = cv2.waitKey(10) & 0xff  # cıkmak icin esc basın.
    if k == 27:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

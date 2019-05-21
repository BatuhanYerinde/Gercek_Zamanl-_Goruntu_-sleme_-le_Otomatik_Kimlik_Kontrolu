import cv2
import numpy as np
from PIL import Image
import os

# veri setlerinin bulunduğu yolun ismi değişkene atılıyor.
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create() #egitim icin nesne olusturuluyor.
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml"); # yuzun on kısmını algılamak ıcın cascade sınıflandırıcısından nesne tanımlanıyor.""

# resimler ve etiketleri veriseti icerisinden bulunup idleri ile birlikte bu fonksiyonda kaydediliyor.
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # Resim gray levela ceviriliyor.
        img_numpy = np.array(PIL_img,'uint8') #resim numpy array biciminde kaydediliyor.

        id = int(os.path.split(imagePath)[-1].split(".")[1]) # idler dosya isimlerinden okunuyor.
        faces = detector.detectMultiScale(img_numpy) #yuzlerın cercevesı algılanarak faces isimli listeye atılıyor.

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w]) # cerceve koordinatları numpy diziye ekleniyor.
            ids.append(id) idler diziye kaydediliyor.

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path) #resim id kaydedici fonksiyon cagırılıyor.
recognizer.train(faces, np.array(ids)) # cerceve ve ide parametrelerine  gore egitim gercekleştiriliyor.

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # egitim dosyası trainer adlı klasor altına kaydediliyor.

#kac adet yuz egitildigi ekranda gosteriliyor.
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

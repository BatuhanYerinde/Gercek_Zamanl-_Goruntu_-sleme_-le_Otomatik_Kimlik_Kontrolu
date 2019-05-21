import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # video genisligi
cam.set(4, 480) # video yukseklıgı

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# her tanıtılan yuz ıcın klavyeden bir numara girilir
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# resim sayısını tutmak icin kullanılan degısken
count = 0

while(True):

    ret, img = cam.read()
    img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # veriseti klasoru icerisine face_id si ve count'u belirtilen resimler kaydediliyor.
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # cıkmak icim ESC tusuna basın.
    if k == 27:
        break
    elif count >= 30: # videodan 30 adet resim cekiliyor daha sonra donguden cıkılıyor.
         break

# program cıkisi
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()



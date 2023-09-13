import cv2
import numpy as np
from PIL import Image
import os
import tkinter as tk
from threading import Thread

# Chemin vers le dossier contenant les données des visages
path = "C:\\Users\\Tshala Benjamin\\Desktop\\videos\\photos"

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
detector = cv2.CascadeClassifier("C:\\Users\\Tshala Benjamin\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml");

names = ['None', 'User1', 'User2', 'User3', 'User4', 'User5']  # TODO: Change to your own user names

# fonction pour obtenir les images et les étiquettes de données
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convertit en niveaux de gris
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Training")
        self.btn_start = tk.Button(self.window, text="Start Training", command=self.start_training)
        self.btn_start.pack(fill='both')
        self.btn_open_camera = tk.Button(self.window, text="Open Camera", command=self.open_camera)
        self.btn_open_camera.pack(fill='both')
        self.lbl_status = tk.Label(self.window, text="Status")
        self.lbl_status.pack(fill='both')

    def start_training(self):
        self.thread1 = Thread(target=self.train_faces)
        self.thread1.start()

    def open_camera(self):
        self.thread2 = Thread(target=self.run_camera)
        self.thread2.start()

    def train_faces(self):
        print ("\n [INFO] Entrainement des visages. Attendez s'il vous plaît...")
        faces,ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # Sauvegarde du modèle dans trainer.yml
        recognizer.write('trainer.yml') 

        print("\n [INFO] {0} visages entrainés. Sortie du programme".format(len(np.unique(ids))))

    def run_camera(self):
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height

        minW = 0.1*cam.get(3)
        minH = 0.1*cam.get(4)

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = detector.detectMultiScale( 
                gray,     
                scaleFactor = 1.2,
                minNeighbors = 5,    
                minSize = (int(minW), int(minH)),
                )

            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

                # If confidence is less than 100, and id is 1 (User1), print matching image and change the color of lbl_status to green
                if (confidence < 100) and (id == 1):
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                    print("Matching Image")
                    self.lbl_status.config(bg='green')

                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

            cv2.imshow('camera',img) 

            k = cv2.waitKey(10) & 0xff 
            if k == 27:
                break

        print("\n [INFO] Exiting Program")
        cam.release()
        cv2.destroyAllWindows()

root = tk.Tk()
app = App(root)
root.mainloop()

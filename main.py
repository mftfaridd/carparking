import cv2
import pickle
import numpy as np
import time
import datetime
import pyrebase
# import firebase_admin
# from firebase_admin import credentials, firestore
from cekmaling import prediksimaling

net = cv2.dnn.readNetFromDarknet("yolov4-tiny-custom_mobil.cfg","yolov4-tiny-custom_final_mobil.weights")

classes = ['mobil']

# Video feed
cap = cv2.VideoCapture(1)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

prev_frame_time = 0 
new_frame_time = 0

with open('CarParkPos', 'rb') as f:
     posList = pickle.load(f)
    
with open('CarParkPos2', 'rb') as f:
    posList2 = pickle.load(f)

firebaseConfig= {
    "apiKey": "AIzaSyAHITI7sMlrEDdtnncGKDjH4N2xBKhqSJ4",
    "authDomain": "web-parkir.firebaseapp.com",
    "databaseURL": "https://web-parkir-default-rtdb.firebaseio.com",
    "projectId": "web-parkir",
    "storageBucket": "web-parkir.appspot.com",
    "messagingSenderId": "931956450541",
    "appId": "1:931956450541:web:a0b1053ecd94f7afd818e4",
    "measurementId": "G-8Z19WV72JW" 
}


# storage = firebase.storage()

# cred = credentials.Certificate("web-parkir-firebase-adminsdk-gwjv3-edcaa1e7c2.json")
# firebase_admin.initialize_app(cred)

# firestore.client()

def tanggal():
    x = datetime.datetime.now()
    time = str(x.strftime("%X"))
    date = str(x.year)
    date += "-"
    date += str(x.month)
    date += "-"
    date += str(x.day)
    date += " "
    date += time
    return date

def predikslot(imgCrop):
    global arr
    global j
    global prev_frame_time
    global new_frame_time
    w, h = 160, 160
    hight, width, _ = imgCrop.shape
    blob = cv2.dnn.blobFromImage(imgCrop, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_name = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_name)
    boxes = []
    confidences = []
    class_ids = []
    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.9:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3] * hight)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, .8, .4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]]) 
            if (label == "mobil"):
                arr[j] = 1
                

            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(imgCrop, (x, y), (x + w, y + h), color, 2)
            cv2.putText(imgCrop, label + " " + confidence, (x, y + 100), font, 2, color, 2)
            cv2.putText(imgCrop, fps, (x,y+200), font, 2, color, 2)
    return imgCrop


def checkParkingSpace(imgPro):
    global arr
    global prev_frame_time
    global new_frame_time
    w, h = 145, 145
    global j
    j = 0
    for pos in posList:
        p, q = pos
        w, h = 140, 140
        imgBg = cv2.imread('white.png')
        imgBg = cv2.resize(imgBg, (640, 480))
        imgCropped = imgPro[q:q + h, p:p + w]
        imgBg[q:q + h, p:p + w] = imgCropped
        predikslot(imgBg)
        cv2.imshow(str(j), imgBg)
       
        
        #print (arr[0],arr[1],arr[2],arr[3],arr[4],arr[5])
#        print ("slot ke-",j," ",arr[j])
        j = j+1
    
#    for pos in posList2:
#        p, q = pos
#        w, h = 70, 50
#        imgBg = cv2.imread('white.png')
#        imgBg = cv2.resize(imgBg, (960, 720))
#        imgCropped = imgPro[q:q + h, p:p + w]
#        imgBg[q:q + h, p:p + w] = imgCropped
#        cv2.imwrite(str(j)+".jpg", imgBg)
#        imgSlot = cv2.imread(str(j)+".jpg")
        
#        imgCrop = predikslot(imgSlot)
#        cv2.imshow(str(j), imgCrop)
       
#        #print ("slot ke-",j," ",arr[j])
#        j = j+1
        
    firebase = pyrebase.initialize_app(firebaseConfig)
    database = firebase.database()
    data = {"Slot1": arr[0], "Slot2": arr[1], "Slot3": arr[2], "Slot4": arr[3], "Slot5": arr[4], "Slot6": arr[5], "Slot7": arr[6], "Slot8": arr[7]}
    database.child("data").set(data)
    print (arr[0],arr[1],arr[2],arr[3],arr[4],arr[5],arr[6])
        
    

    


 
while True:
 
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()
    global arr
    arr = [0,0,0,0,0,0,0,0]
    img = cv2.resize(img, (640, 480))
    prediksimaling(img)
    checkParkingSpace(img)
    # cv2.imshow("image",img)
    cv2.waitKey(1)
    
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

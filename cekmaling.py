import cv2
import pickle
import numpy as np
import time
import datetime
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore

net = cv2.dnn.readNetFromDarknet("yolov4-tiny-custom.cfg","yolov4-tiny-custom_final.weights")

classes = ['maling', 'normal']



prev_frame_time = 0 
new_frame_time = 0


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

firebase = pyrebase.initialize_app(firebaseConfig)
database = firebase.database()
storage = firebase.storage()

cred = credentials.Certificate("web-parkir-firebase-adminsdk-gwjv3-edcaa1e7c2.json")
firebase_admin.initialize_app(cred)

firestore.client()

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

def prediksimaling(imgCrop):
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
                
            if (label == 'maling' ):    
                waktu = tanggal()
                cv2.imwrite(waktu+".jpg", imgCrop)
                savingimage = waktu+".jpg"
                storage.child("gambar/"+waktu+".jpg").put(savingimage)
            
                url = storage.child("gambar/"+waktu+".jpg").get_url('token') 
                dataa = {"url": url, "tanggal": waktu}
                db.collection('gambar').add(dataa)
                
            
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(imgCrop, (x, y), (x + w, y + h), color, 2)
            cv2.putText(imgCrop, label + " " + confidence, (x, y + 100), font, 2, color, 2)
            cv2.putText(imgCrop, fps, (x,y+200), font, 2, color, 2)
    return imgCrop




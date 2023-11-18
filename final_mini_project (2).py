#!/usr/bin/env python
# coding: utf-8

# In[108]:


import numpy as np 
import pandas as pd 
import os
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import dlib
from scipy.spatial import distance as dist

from imutils import face_utils
import numpy as np
import argparse
import imutils
import time


# In[2]:


os.chdir("C:\\Users\\abhis\\PycharmProjects\\MiniProject")


# In[3]:


labels=os.listdir("train")


# In[4]:


labels


# In[5]:


def face_for_yawn(direc="train", face_cas_path="haarcascade_frontalface_default.xml"):
    yaw_no = []
    IMG_SIZE = 145
    categories = ["yawn", "no_yawn"]
    for category in categories:
        path_link = os.path.join(direc, category)
        class_num1 = categories.index(category)
        print(class_num1)
        for image in os.listdir(path_link):
            image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
            face_cascade = cv2.CascadeClassifier(face_cas_path)
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
            for (x, y, w, h) in faces:
                img = cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_color = img[y:y+h, x:x+w]
                resized_array = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
                yaw_no.append([resized_array, class_num1])
    return yaw_no

yawn_no_yawn = face_for_yawn()


# In[6]:


def get_data(dir_path="train", face_cas="haarcascade_frontalface_default.xml", eye_cas="haarcascade.xml"):
    labels = ['Closed', 'Open']
    IMG_SIZE = 145
    data = []
    for label in labels:
        path = os.path.join(dir_path, label)
        class_num = labels.index(label)
        class_num +=2
        print(class_num)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array, class_num])
            except Exception as e:
                print(e)
    return data

data_train = get_data()


# In[7]:


def append_data():
    yaw_no = face_for_yawn()
    data = get_data()
    yaw_no.extend(data)
    return np.array(yaw_no)


# In[8]:


new_data = append_data()


# In[9]:


X = []
y = []
for feature, label in new_data:
    X.append(feature)
    y.append(label)


# In[10]:


X = np.array(X)
X = X.reshape(-1, 145, 145, 3)


# In[12]:


y = np.array(y)


# In[13]:


from sklearn.model_selection import train_test_split
seed = 42
test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)


# In[14]:


len(X_test)


# In[17]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


# In[18]:


train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
test_generator = ImageDataGenerator(rescale=1/255)

train_generator = train_generator.flow(np.array(X_train), y_train, shuffle=False)
test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)


# In[19]:


model = Sequential()

model.add(Conv2D(256, (3, 3), activation="relu", input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(64, activation="relu"))
model.add(Dense(4, activation="softmax"))

model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")

model.summary()


# In[37]:


history = model.fit(train_generator, epochs=50, validation_data=test_generator, shuffle=True, validation_steps=len(test_generator))


# In[4]:


model.save("drowsiness_new6.h5")


# In[7]:


model.save("drowsiness_new6.model")


# In[21]:


testpredict = np.argmax(model.predict(X_test), axis=-1)


# In[20]:


model=load_model('drowsiness_new6.h5')


# In[22]:


testpredict


# In[37]:


labels_new = ["yawn", "no_yawn", "Closed", "Open"]


# In[42]:


IMG_SIZE = 145
def prepare(filepath, face_cas="haarcascade_frontalface_default.xml"):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_array = img_array / 255
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
model = tf.keras.models.load_model("drowsiness_new6.h5")


# In[43]:


prediction = model.predict([prepare("train/no_yawn/147.jpg")])
np.argmax(prediction)


# In[44]:


prediction = model.predict([prepare("train/yawn/71.jpg")])
np.argmax(prediction)


# In[18]:


prediction = model.predict([prepare("train/Closed/_101.jpg")])
np.argmax(prediction)


# In[28]:


prediction = model.predict([prepare("train/Open/_104.jpg")])
np.argmax(prediction)


# In[29]:


test_img=cv2.imread("train\open\_2.jpg")
test_img=test_img/255
resizedarray=cv2.resize(test_img,(145,145))
testnew_array = resizedarray.reshape(-1, 145,145, 3)
prediction = model.predict(testnew_array)
print(np.argmax(prediction))


# In[30]:


test_img=cv2.imread("train/yawn/10.jpg",cv2.IMREAD_COLOR)
test_img=test_img/255
resized_array = cv2.resize(test_img, (145, 145))
testnew_array = resizedarray.reshape(-1, 145,145, 3)
prediction = model.predict(testnew_array)
print(np.argmax(prediction))


# In[60]:


img=cv2.imread("kajal.jpg")

face_casacde= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
eyes=eye_cascade.detectMultiScale(gray,1.1,4)
for x,y,w,h in eyes:

    cv2.rectangle(img, (x, y), (x + w +5, y + h + 5), (0, 255, 0), 3)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[61]:


for x,y,w,h in eyes:
    roi_gray=gray[y:y+h,x:x+w]
    roi_color=img[y:y+h,x:x+w]
    eyess=eye_cascade.detectMultiScale(roi_gray)
    if(len(eyess)==0):
        print("eyes not detected")
    else:
        for ex,ey,ew,eh in eyess:
            eyess_roi=roi_color[ey:ey+eh,ex:ex+ew]


# In[62]:


plt.imshow(cv2.cvtColor(eyes_roi,cv2.COLOR_BGR2RGB))


# In[63]:


eyes_roi.shape


# In[64]:


resized_img=cv2.resize(eyes_roi,(145,145))
final_img=resized_img.reshape(-1,145,145,3)
final_img=final_img/255
print(np.argmax(model.predict(final_img)))


# In[105]:



face_casacde = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap=cv2.VideoCapture(0)
while True:
    sucess,frame = cap.read()
    try:
        frame = cv2.resize(frame,(300, 300))
    except:
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    eyes=eye_cascade.detectMultiScale(gray,1.1,4)
    for (x, y, w, h) in eyes:
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0),2)
        eyess=eye_cascade.detectMultiScale(roi_gray)
        if len(eyess)==0:
            print("eyes are not detected")
        else:
            for(ex,ey,ew,eh) in eyess:
                eyess_roi=roi_color[ey:ey+eh,ex:ex+ew]
                
    resized_img=cv2.resize(eyess_roi,(145,145))
    final_img=resized_img.reshape(-1,145,145,3)
    final_image=final_img/255
    predictions=np.argmax(model.predict(final_image))
    if(predictions==2):
        status = "Drowsy"   
    elif(predictions == 3):
        status="Active"
    else:
        status ="not Detected"
        
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_casacde.detectMultiScale(gray, 1.3, 3)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w , y + h + 20), (0, 255, 0), 2)
    
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,status,(30,50),font,2,(255,0,0),2,cv2.LINE_4)
    cv2.imshow("drowsiness",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
        
cap.release()
    


# In[ ]:





# In[103]:


counter=0

face_casacde = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap=cv2.VideoCapture(0)
while True:
    sucess,frame = cap.read()
    try:
        frame = cv2.resize(frame,(300, 300))
    except:
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    eyes=eye_cascade.detectMultiScale(gray,1.1,4)
    for (x, y, w, h) in eyes:
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0),2)
        eyess=eye_cascade.detectMultiScale(roi_gray)
        if len(eyess)==0:
            print("eyes are not detected")
        else:
            for(ex,ey,ew,eh) in eyess:
                eyess_roi=roi_color[ey:ey+eh,ex:ex+ew]
                
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_casacde.detectMultiScale(gray, 1.3, 3)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w , y + h + 20), (0, 255, 0), 2)
    
    font=cv2.FONT_HERSHEY_COMPLEX
                
    resized_img=cv2.resize(eyess_roi,(145,145))
    final_img=resized_img.reshape(-1,145,145,3)
    final_image=final_img/255
    predictions=np.argmax(model.predict(final_image))
    if(predictions==3):
        status = "opened"
        cv2.putText(frame,status,(50,50),font,1,(0,255,0),2,cv2.LINE_4)
        x1,y1,w1,h1=0,250,175,50
        
        cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,0,0),-1)
        
        cv2.putText(frame,'Active',(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)
    else:
        counter=counter+1
        status="closed "
        cv2.putText(frame,status,(50,50),font,1,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x, y), (x + w , y + h + 20), (0, 255, 0), 2)
        if counter>20:
            x1,y1,w1,h1=0,250,175,75
            cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,0,0),-1)
            cv2.putText(frame,'Drowsy',(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
            while(counter>=0):
                counter = counter - 1 
                cv2.putText(frame,'Drowsy',(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
           
    cv2.imshow("drowsiness",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
        
cap.release()
    


# In[110]:


YAWN_THRESH = 20


# In[109]:


def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


# In[119]:


predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_casacde = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# In[140]:


count=0
counter = 0


cap=cv2.VideoCapture(0)
while True:
    sucess,frame = cap.read()
    try:
        frame = cv2.resize(frame,(300, 300))
    except:
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    eyes=eye_cascade.detectMultiScale(gray,1.1,4)
    
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in eyes:
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0),2)
        eyess=eye_cascade.detectMultiScale(roi_gray)
        if len(eyess)==0:
            print("eyes are not detected")
        else:
            for(ex,ey,ew,eh) in eyess:
                eyess_roi=roi_color[ey:ey+eh,ex:ex+ew]
                
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
   #faces = face_casacde.detectMultiScale(gray, 1.3, 3)
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w , y + h + 20), (0, 255, 0), 2)
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        distance = lip_distance(shape)
        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0,255, 0), 2)
        if (distance > YAWN_THRESH):
            x2,y2,w2,h2 = 175,250,125,75
            cv2.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(0,0,0),-1)
            cv2.putText(frame, "Yawn Alert",(x2+int(w2/10),y2+int(h2/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    
    font=cv2.FONT_HERSHEY_COMPLEX
                
    resized_img=cv2.resize(eyess_roi,(145,145))
    final_img=resized_img.reshape(-1,145,145,3)
    final_image=final_img/255
    predictions=np.argmax(model.predict(final_image))
    
    
    
    
    
    if(predictions==3):
        status = "opened"
        cv2.putText(frame,status,(50,50),font,1,(0,255,0),2,cv2.LINE_4)
        x1,y1,w1,h1=0,250,175,50
        
        cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,0,0),-1)
        
        cv2.putText(frame,'Active',(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)
    else:
        count=count+1
        status="closed "
        cv2.putText(frame,status,(50,50),font,1,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x, y), (x + w , y + h + 20), (0, 255, 0), 2)
        if count>20:
            x1,y1,w1,h1=0,250,175,75
            cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,0,0),-1)
            cv2.putText(frame,'Drowsy',(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
            while(count>=0):
                count = count - 1 
                cv2.putText(frame,'Drowsy',(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
           
    cv2.imshow("drowsiness",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
        
cap.release()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt


# In[29]:


os.chdir("C:\\Users\\vetsa\\PycharmProjects\\pythonProject\\Opencvtest")


# In[30]:


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


# In[4]:


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


# In[5]:


data_train = get_data()


# In[6]:


def append_data():
    yaw_no = face_for_yawn()
    data = get_data()
    yaw_no.extend(data)
    return np.array(yaw_no)


# In[7]:


new_data = append_data()


# In[8]:


X = []
y = []
for feature, label in new_data:
    X.append(feature)
    y.append(label)


# In[9]:


X = np.array(X)
X = X.reshape(-1, 145, 145, 3)


# In[10]:


from sklearn.preprocessing import LabelBinarizer
label_bin = LabelBinarizer()
y = label_bin.fit_transform(y)


# In[11]:


y = np.array(y)


# In[12]:


from sklearn.model_selection import train_test_split
seed = 42
test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)


# In[13]:


len(X_test)
X_train.shape


# In[14]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.optimizers import Adam


# In[15]:


tf.__version__


# In[16]:


train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
test_generator = ImageDataGenerator(rescale=1/255)

train_generator = train_generator.flow(np.array(X_train), y_train, shuffle=False)
test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)


# In[17]:


model = Sequential()



model.add(Conv2D(64, (3, 3), activation="relu",input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())



model.add(Dense(4, activation="softmax"))

model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=Adam(learning_rate=0.1))

model.summary()


# In[18]:


history = model.fit(train_generator, epochs=5, validation_data=test_generator, shuffle=True, validation_steps=len(test_generator))


# In[19]:


model.save("cnn1model.h5")


# In[20]:


model=load_model('cnn1model.h5')


# In[21]:


testpredict = np.argmax(model.predict(X_test), axis=-1)


# In[22]:


testpredict


# In[23]:


labels_new = ["yawn", "no_yawn", "Closed", "Open"]
IMG_SIZE = 145
def prepare(filepath, face_cas="haarcascade_frontalface_default.xml"):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_array = img_array / 255
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

model = tf.keras.models.load_model("cnn1model.h5")


# In[24]:


prediction = model.predict([prepare("train/Closed/_101.jpg")])
np.argmax(prediction)


# In[25]:


prediction = model.predict([prepare("train/Open/_104.jpg")])
np.argmax(prediction)


# In[26]:


test_img=cv2.imread("train\open\_2.jpg")
test_img=test_img/255
resizedarray=cv2.resize(test_img,(145,145))
testnew_array = resizedarray.reshape(-1,145,145, 3)
prediction = model.predict(testnew_array)
print(np.argmax(prediction))


# In[ ]:


face_casacde= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 3)
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
        status="closed eyes"
    else:
        status="open eyes"
        
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_casacde.detectMultiScale(gray, 1.3, 3)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,status,(50,50),font,3,(255,0,0),2,cv2.LINE_4)
    cv2.imshow("drowsiness",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
        
cap.release()
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





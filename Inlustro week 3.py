#!/usr/bin/env python
# coding: utf-8

# # Handwritten digit recognition

# # ANN

# In[1]:


pip install tensorflow


# In[2]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model,Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.utils import to_categorical,plot_model
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
from warnings import filterwarnings
filterwarnings('ignore')


# In[3]:


(x_train,y_train),(x_test,y_test)=mnist.load_data()
print("Training set size:" , x_train.shape,y_train.shape)
print("testing set size:" , x_test.shape,y_test.shape)


# In[4]:


#no of classes

num_labels= len(np.unique(y_train))
num_labels


# In[5]:


plt.figure(figsize=(5,5))
plt.imshow(x_train[900],cmap='gray') #imshow---image show


# In[6]:


plt.figure(figsize=(5,5))
for i in range(0,7):
    ax=plt.subplot(5,5,i+1)
    plt.imshow(x_train[i],cmap='gray')
    plt.axis('off')


# In[7]:


def visualize_img(data,num=10):
    plt.figure(figsize=(5,5))
    for i in range(0,num):
        ax=plt.subplot(5,5,i+1)
        plt.imshow(data[i],cmap='gray')
        plt.axis('off')


# In[8]:


visualize_img(x_train,15)


# In[9]:


def pixel_visualize(img):
    fig=plt.figure(figsize=(12,12))
    ax=fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
    width,height=img.shape
    
    threshold=img.max()/2.5
    
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)),xy=(y,x),
                        color='white' if img[x][y]<threshold else "black")


# In[10]:


pixel_visualize(x_train[2])


# # DATA PREPARATION

# In[11]:


#encoding for dependent variable


y_train[0:5]


# In[12]:


y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


# In[13]:


y_train[0:5]


# # reshaping

# In[14]:


image_size=x_train.shape[1]
image_size


# In[15]:


print(f"x_train size:{x_train.shape}\n\nx_test size:{x_test.shape}")


# In[16]:


x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)


# In[17]:


print(f"x_train size:{x_train.shape}\n\nx_test size:{x_test.shape}")


# In[18]:


x_train=x_train.astype("float32")/255
x_test=x_test.astype("float32")/255


# # Modelling

# In[19]:



model = tf.keras.Sequential([
    Flatten(input_shape=(28,28,1)), # making the data understandable
    Dense(units=128, activation="relu",name="layer1"),  # hidden layer
    Dense(units=num_labels, activation="softmax",name="output_layer")])

model.compile(loss="categorical_crossentropy",  # error evaluation metric ,
    optimizer="adam", # optimization algorithm to minimize the loss function
    metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),"ACCURACY"])


# In[20]:


model.summary()


# In[21]:


#Model fit

model.fit(x_train,y_train,epochs=8,batch_size=128,
          validation_data=(x_test,y_test))


# In[22]:


history= model.fit(x_train,y_train,epochs=8,batch_size=128, #saving the model
          validation_data=(x_test,y_test))


# In[23]:


loss,precision,recall,acc=model.evaluate(x_test,y_test,verbose=False)
print(f"Test accuracy:{round(acc*100,2)}")
print(f"Test loss:{round(loss*100,2)}")
print(f"Test precision:{round(precision*100,2)}")
print(f"Test recall:{round(recall*100,2)}")


# # Prediction & visualization

# In[24]:


y_pred=model.predict(x_test)


# In[25]:


y_pred_classes=np.argmax(y_pred,axis=1)


# In[26]:


if len(y_test.shape)> 1 and y_test.shape[1]!= 1:
    y_test=np.argmax(y_test,axis=1)


# In[27]:


#confusion matrix

cm=confusion_matrix(y_test,y_pred_classes)
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True,fmt='d',cmap='Reds')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('confusion matrix')
plt.show()


# # Registering model

# In[28]:


model.save("mnist_model.h5")


# In[29]:


import random
random=random.randint(0,x_test.shape[0])
random


# In[30]:


test_image=x_test[random]


# In[31]:


y_test[random]


# In[32]:


plt.figure(figsize=(5,5))
plt.imshow(test_image.reshape(28,28),cmap='gray')


# In[33]:


test_data=x_test[random].reshape(1,28,28,1)


# In[34]:


probability=model.predict(test_data)


# In[35]:


predicted_classes=np.argmax(probability)


# In[36]:


print(f"predicted class:{predicted_classes}\nProbabilty Value of Other Classes: {probability}")


# # Convolutional neural network

# # Object detection

# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, GlobalAveragePooling2D, UpSampling2D, Input
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import scipy as np


# In[41]:


(train_data,train_labels),(test_data, test_labels)= tf.keras.datasets.cifar10.load_data()


# In[42]:


train_data.shape, train_labels.shape, train_data.dtype, test_data.dtype


# In[46]:


class_names=['airplane','automobile','bird','horse','cat','deer','dog','frog','ship','truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_data[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
    


# In[47]:


train_data=train_data.astype(np.float32)
test_data=test_data.astype(np.float32)
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)
train_data/=255
test_data/=255


# In[48]:


inp=Input(shape=(32,32,3))
x=Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same')(inp)
x=Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same')(x)
x=MaxPooling2D(pool_size=2,strides=2,padding='valid')(x)
x=Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same')(x)
x=Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same')(x)
x=MaxPooling2D(pool_size=2,strides=2,padding='valid')(x)
x=Flatten()(x)
x=Dropout(0.4)(x)
x=Dense(units=64,activation='relu')(x)
x=Dense(units=10,activation='softmax')(x)
model_costume_cnn=Model(inp,x)
model_costume_cnn.summary()


# In[51]:


model_costume_cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history_costume_cnn=model_costume_cnn.fit(train_data,train_labels,batch_size=8,epochs=3,validation_split=0.15)


# #  Resnet50 for finetuning

# In[56]:


(train_data,train_labels),(test_data,test_labels)=tf.keras.datasets.cifar10.load_data()
train_data=tf.keras.applications.resnet50.preprocess_input(train_data)
test_data=tf.keras.applications.resnet50.preprocess_input(test_data)


# In[59]:


resnet=tf.keras.applications.ResNet50(input_shape=(224,224,3),include_top=False,weights='imagenet',classes=10)
resnet.trainable=False
inputs=Input((32,32,3))
x=UpSampling2D((7,7))(inputs)
x=resnet(x)
x=GlobalAveragePooling2D()(x)
x=Dropout(0.3)(x)
x=Dense(units=10,activation='relu'(x))
x=BatchNormalization()(x)
output=Dense(units=10,activation='softmax'())
model_resnet=Model(inputs,output)
model_resnet.summary()


# In[ ]:


resnet_learning_rate=0.001
model_resnet.compile(optimizer=Adam(learning))


# In[ ]:





# In[ ]:





# In[ ]:





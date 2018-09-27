
# coding: utf-8

# In[5]:


from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import os
import numpy as np
from keras.layers import merge, Input, Lambda
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
import pickle
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, GlobalAveragePooling2D
from keras.models import Model, Input
from keras.utils import to_categorical
import os
import numpy as np
import pandas as pd
import keras
from pandas import DataFrame
from keras import models
from keras.layers import Conv1D, Concatenate, merge
from keras.models import load_model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import keras,pickle
from keras.layers import Activation, Dense, InputLayer, MaxPool1D, Flatten, Dropout, AvgPool1D, Input, concatenate, Concatenate, Add, Reshape
from keras.utils import np_utils
from keras.utils import np_utils
from keras.layers.merge import Concatenate
from keras.models import Model
np.random.seed(10)
from tensorflow import set_random_seed
set_random_seed(15)
from pprint import pprint
from keras.layers.embeddings import Embedding
from numpy import newaxis
from keras import backend as K
# from keras.utils import plot_model


# In[22]:




def my_simple_cnn(inputs) :
    x = Conv2D(32, (3,3), activation='relu')(inputs)
    x = AveragePooling2D()(x)
    x = Conv2D(32, (1,1), activation='relu')(x)
    x = Conv2D(16, (3,3), activation='relu')(x)
    x = AveragePooling2D()(x)
    x = Conv2D(16, (3,3), activation='relu')(x)
    x = Conv2D(16, (3,3), activation='relu')(x)
    x = AveragePooling2D()(x)
    return x





input1 = Input(shape=(224,224, 3))
input2 = Input(shape=(224,224, 3))
input3 = Input(shape=(224,224, 3))
input4 = Input(shape=(224,224, 3))
input5 = Input(shape=(224,224, 3))
input6 = Input(shape=(224,224, 3))
input7 = Input(shape=(224,224, 3))
input8 = Input(shape=(224,224, 3))
input9 = Input(shape=(224,224, 3))
input10 = Input(shape=(224,224, 3))
input11 = Input(shape=(224,224, 3))
input12 = Input(shape=(224,224, 3))
input13 = Input(shape=(224,224, 3))
input14 = Input(shape=(224,224, 3))
input15 = Input(shape=(224,224, 3))
input16 = Input(shape=(224,224, 3))
input17 = Input(shape=(224,224, 3))
input18 = Input(shape=(224,224, 3))
input19 = Input(shape=(224,224, 3))
input20 = Input(shape=(224,224, 3))
input21 = Input(shape=(224,224, 3))
input22 = Input(shape=(224,224, 3))
input23 = Input(shape=(224,224, 3))
input24 = Input(shape=(224,224, 3))
input25 = Input(shape=(224,224, 3))
input26 = Input(shape=(224,224, 3))
input27 = Input(shape=(224,224, 3))
input28 = Input(shape=(224,224, 3))
input29 = Input(shape=(224,224, 3))
input30 = Input(shape=(224,224, 3))

output1 = my_simple_cnn(input1)
output2 = my_simple_cnn(input2)
output3 = my_simple_cnn(input3)
output4 = my_simple_cnn(input4)
output5 = my_simple_cnn(input5)
output6 = my_simple_cnn(input6)
output7 = my_simple_cnn(input7)
output8 = my_simple_cnn(input8)
output9 = my_simple_cnn(input9)
output10 = my_simple_cnn(input10)
output11 = my_simple_cnn(input11)
output12 = my_simple_cnn(input12)
output13 = my_simple_cnn(input13)
output14 = my_simple_cnn(input14)
output15 = my_simple_cnn(input15)
output16 = my_simple_cnn(input16)
output17 = my_simple_cnn(input17)
output18 = my_simple_cnn(input18)
output19 = my_simple_cnn(input19)
output20= my_simple_cnn(input20)
output21 = my_simple_cnn(input21)
output22 = my_simple_cnn(input22)
output23 = my_simple_cnn(input23)
output24 = my_simple_cnn(input24)
output25 = my_simple_cnn(input25)
output26 = my_simple_cnn(input26)
output27 = my_simple_cnn(input27)
output28 = my_simple_cnn(input28)
output29 = my_simple_cnn(input29)
output30 = my_simple_cnn(input30)


my_inputs = [input1,input2,input3,input4,input5,input6,
    input7,input8,input9,input10,input11,input12,input13,input14,input15,input16,input17,input18,input19,input20,
    input21,input22,input23,input24,input25,input26,input27,input28,input29,input30]


my_outputs = [output1,output2,output3,output4,output5,output6,
    output7,output8,output9,output10,output11,output12,output13,output14,
    output15,output16,output17,output18,output19,output20,output21,output22,
    output23,output24,output25,output26,output27,output28,output29,output30]
concat_output = Concatenate(axis = -1)(my_outputs)



predictions = Flatten()(concat_output)
predictions = Dense(300, activation='relu')(predictions)
predictions = Dense(200, activation='relu')(predictions)
predictions = Dense(100, activation='relu')(predictions)

predictions = Dense(2, activation='softmax')(predictions)

from keras.optimizers import SGD
opt = SGD(lr=0.001)

model = Model(inputs=my_inputs, outputs=predictions)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
# In[23]:


from glob import glob
from random import shuffle

pat_list0 = glob('C:/Users/korea/Desktop/sungmo_RESNET/data/30image/0/*')
pat_list1 = glob('C:/Users/korea/Desktop/sungmo_RESNET/data/30image/1/*')

from pprint import pprint
######1
file_list0_1 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_1.append(i + "/" + str(tmp_list[0]))
pprint(file_list0_1)

file_list1_1 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_1.append(i + "/" + str(tmp_list[0]))
pprint(file_list1_1)

######2
file_list0_2 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_2.append(i + "/" + str(tmp_list[1]))
pprint(file_list0_2)

file_list1_2 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_2.append(i + "/" + str(tmp_list[1]))
pprint(file_list1_2)



######3
file_list0_3 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_3.append(i + "/" + str(tmp_list[2]))
pprint(file_list0_3)

file_list1_3 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_3.append(i + "/" + str(tmp_list[2]))
pprint(file_list1_3)


######4
file_list0_4 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_4.append(i + "/" + str(tmp_list[3]))
pprint(file_list0_4)

file_list1_4 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_4.append(i + "/" + str(tmp_list[3]))
pprint(file_list1_4)

######5
file_list0_5 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_5.append(i + "/" + str(tmp_list[4]))
pprint(file_list0_5)

file_list1_5 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_5.append(i + "/" + str(tmp_list[4]))
pprint(file_list1_5)

######6
file_list0_6 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_6.append(i + "/" + str(tmp_list[5]))
pprint(file_list0_6)

file_list1_6 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_6.append(i + "/" + str(tmp_list[5]))
pprint(file_list1_6)

######7
file_list0_7 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_7.append(i + "/" + str(tmp_list[6]))
pprint(file_list0_7)

file_list1_7 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_7.append(i + "/" + str(tmp_list[6]))
pprint(file_list1_7)

######8
file_list0_8 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_8.append(i + "/" + str(tmp_list[7]))
pprint(file_list0_8)

file_list1_8 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_8.append(i + "/" + str(tmp_list[7]))
pprint(file_list1_8)

######9
file_list0_9 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_9.append(i + "/" + str(tmp_list[8]))
pprint(file_list0_9)

file_list1_9 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_9.append(i + "/" + str(tmp_list[8]))
pprint(file_list1_9)

######10
file_list0_10 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_10.append(i + "/" + str(tmp_list[9]))
pprint(file_list0_10)

file_list1_10 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_10.append(i + "/" + str(tmp_list[9]))
pprint(file_list1_10)


######11
file_list0_11 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_11.append(i + "/" + str(tmp_list[10]))
pprint(file_list0_11)

file_list1_11 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_11.append(i + "/" + str(tmp_list[10]))
pprint(file_list1_11)


######12
file_list0_12 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_12.append(i + "/" + str(tmp_list[11]))
pprint(file_list0_12)

file_list1_12 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_12.append(i + "/" + str(tmp_list[11]))
pprint(file_list1_12)


######13
file_list0_13 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_13.append(i + "/" + str(tmp_list[12]))
pprint(file_list0_13)

file_list1_13 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_13.append(i + "/" + str(tmp_list[12]))
pprint(file_list1_13)


######14
file_list0_14 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_14.append(i + "/" + str(tmp_list[13]))
pprint(file_list0_14)

file_list1_14 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_14.append(i + "/" + str(tmp_list[13]))
pprint(file_list1_14)


######15
file_list0_15 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_15.append(i + "/" + str(tmp_list[14]))
pprint(file_list0_15)

file_list1_15 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_15.append(i + "/" + str(tmp_list[14]))
pprint(file_list1_15)


######16
file_list0_16 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_16.append(i + "/" + str(tmp_list[15]))
pprint(file_list0_16)

file_list1_16 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_16.append(i + "/" + str(tmp_list[15]))
pprint(file_list1_16)


######17
file_list0_17 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_17.append(i + "/" + str(tmp_list[16]))
pprint(file_list0_17)

file_list1_17 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_17.append(i + "/" + str(tmp_list[16]))
pprint(file_list1_17)


######18
file_list0_18 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_18.append(i + "/" + str(tmp_list[17]))
pprint(file_list0_18)

file_list1_18 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_18.append(i + "/" + str(tmp_list[17]))
pprint(file_list1_18)


######19
file_list0_19 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_19.append(i + "/" + str(tmp_list[18]))
pprint(file_list0_19)

file_list1_19 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_19.append(i + "/" + str(tmp_list[18]))
pprint(file_list1_19)


######20
file_list0_20 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_20.append(i + "/" + str(tmp_list[19]))
pprint(file_list0_20)

file_list1_20 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_20.append(i + "/" + str(tmp_list[19]))
pprint(file_list1_20)


######21
file_list0_21 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_21.append(i + "/" + str(tmp_list[20]))
pprint(file_list0_21)

file_list1_21 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_21.append(i + "/" + str(tmp_list[20]))
pprint(file_list1_21)

######22
file_list0_22 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_22.append(i + "/" + str(tmp_list[21]))
pprint(file_list0_22)

file_list1_22 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_22.append(i + "/" + str(tmp_list[21]))
pprint(file_list1_22)

######23
file_list0_23 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_23.append(i + "/" + str(tmp_list[22]))
pprint(file_list0_23)

file_list1_23 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_23.append(i + "/" + str(tmp_list[22]))
pprint(file_list1_23)

######24
file_list0_24 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_24.append(i + "/" + str(tmp_list[23]))
pprint(file_list0_24)

file_list1_24 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_24.append(i + "/" + str(tmp_list[23]))
pprint(file_list1_24)

######25
file_list0_25 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_25.append(i + "/" + str(tmp_list[24]))
pprint(file_list0_25)

file_list1_25 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_25.append(i + "/" + str(tmp_list[24]))
pprint(file_list1_25)

######26
file_list0_26 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_26.append(i + "/" + str(tmp_list[25]))
pprint(file_list0_26)

file_list1_26 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_26.append(i + "/" + str(tmp_list[25]))
pprint(file_list1_26)

######27
file_list0_27 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_27.append(i + "/" + str(tmp_list[26]))
pprint(file_list0_27)

file_list1_27 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_27.append(i + "/" + str(tmp_list[26]))
pprint(file_list1_27)

######28
file_list0_28 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_28.append(i + "/" + str(tmp_list[27]))
pprint(file_list0_28)

file_list1_28 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_28.append(i + "/" + str(tmp_list[27]))
pprint(file_list1_28)

######29
file_list0_29 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_29.append(i + "/" + str(tmp_list[28]))
pprint(file_list0_29)

file_list1_29 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_29.append(i + "/" + str(tmp_list[28]))
pprint(file_list1_29)

######30
file_list0_30 = []
for i in pat_list0:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list0_30.append(i + "/" + str(tmp_list[29]))
pprint(file_list0_30)

file_list1_30 = []
for i in pat_list1:
    tmp_list = os.listdir(i)
    tmp_list = sorted(tmp_list, key = int)
    file_list1_30.append(i + "/" + str(tmp_list[29]))
pprint(file_list1_30)













def my_new_axis(x):
    x= x[newaxis,:,:,:]

    return x



# 첫번째 CNN에 들어갈 Input임


input_image_for_cnn_1 = pickle.load(open(file_list0_1[0],'rb'))
input_image_for_cnn_1 = my_new_axis(input_image_for_cnn_1)

for i in range(len(file_list0_1)):

    img = pickle.load(open(file_list0_1[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_1 = np.append(input_image_for_cnn_1, img, axis = 0)


input_image_for_cnn_1 = input_image_for_cnn_1[1:,:,:,:]

for j in range(len(file_list1_1)):

    img = pickle.load(open(file_list1_1[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_1 = np.append(input_image_for_cnn_1, img, axis = 0)

print(input_image_for_cnn_1.shape)



# 두번째 CNN에 들어갈 Input임


input_image_for_cnn_2 = pickle.load(open(file_list0_2[0],'rb'))
input_image_for_cnn_2 = my_new_axis(input_image_for_cnn_2)

for i in range(len(file_list0_2)):

    img = pickle.load(open(file_list0_2[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_2 = np.append(input_image_for_cnn_2, img, axis = 0)


input_image_for_cnn_2 = input_image_for_cnn_2[1:,:,:,:]

for j in range(len(file_list1_2)):

    img = pickle.load(open(file_list1_2[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_2 = np.append(input_image_for_cnn_2, img, axis = 0)


print(input_image_for_cnn_2.shape)


# 세번째 CNN에 들어갈 Input임


input_image_for_cnn_3 = pickle.load(open(file_list0_3[0],'rb'))
input_image_for_cnn_3 = my_new_axis(input_image_for_cnn_3)

for i in range(len(file_list0_3)):

    img = pickle.load(open(file_list0_3[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_3 = np.append(input_image_for_cnn_3, img, axis = 0)


input_image_for_cnn_3 = input_image_for_cnn_3[1:,:,:,:]

for j in range(len(file_list1_3)):

    img = pickle.load(open(file_list1_3[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_3 = np.append(input_image_for_cnn_3, img, axis = 0)


print(input_image_for_cnn_3.shape)


# 4번째 CNN에 들어갈 Input임


input_image_for_cnn_4 = pickle.load(open(file_list0_4[0],'rb'))
input_image_for_cnn_4 = my_new_axis(input_image_for_cnn_4)

for i in range(len(file_list0_4)):

    img = pickle.load(open(file_list0_4[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_4 = np.append(input_image_for_cnn_4, img, axis = 0)


input_image_for_cnn_4 = input_image_for_cnn_4[1:,:,:,:]

for j in range(len(file_list1_4)):

    img = pickle.load(open(file_list1_4[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_4 = np.append(input_image_for_cnn_4, img, axis = 0)


print(input_image_for_cnn_4.shape)


# 5번째 CNN에 들어갈 Input임


input_image_for_cnn_5 = pickle.load(open(file_list0_5[0],'rb'))
input_image_for_cnn_5 = my_new_axis(input_image_for_cnn_5)

for i in range(len(file_list0_5)):

    img = pickle.load(open(file_list0_5[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_5 = np.append(input_image_for_cnn_5, img, axis = 0)


input_image_for_cnn_5 = input_image_for_cnn_5[1:,:,:,:]

for j in range(len(file_list1_5)):

    img = pickle.load(open(file_list1_5[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_5 = np.append(input_image_for_cnn_5, img, axis = 0)


print(input_image_for_cnn_5.shape)

# 6번째 CNN에 들어갈 Input임


input_image_for_cnn_6 = pickle.load(open(file_list0_6[0],'rb'))
input_image_for_cnn_6 = my_new_axis(input_image_for_cnn_6)

for i in range(len(file_list0_6)):

    img = pickle.load(open(file_list0_6[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_6 = np.append(input_image_for_cnn_6, img, axis = 0)


input_image_for_cnn_6 = input_image_for_cnn_6[1:,:,:,:]

for j in range(len(file_list1_6)):

    img = pickle.load(open(file_list1_6[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_6 = np.append(input_image_for_cnn_6, img, axis = 0)


print(input_image_for_cnn_6.shape)

# 7번째 CNN에 들어갈 Input임


input_image_for_cnn_7 = pickle.load(open(file_list0_7[0],'rb'))
input_image_for_cnn_7 = my_new_axis(input_image_for_cnn_7)

for i in range(len(file_list0_7)):

    img = pickle.load(open(file_list0_7[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_7 = np.append(input_image_for_cnn_7, img, axis = 0)


input_image_for_cnn_7 = input_image_for_cnn_7[1:,:,:,:]

for j in range(len(file_list1_7)):

    img = pickle.load(open(file_list1_7[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_7 = np.append(input_image_for_cnn_7, img, axis = 0)


print(input_image_for_cnn_7.shape)

# 8번째 CNN에 들어갈 Input임


input_image_for_cnn_8 = pickle.load(open(file_list0_8[0],'rb'))
input_image_for_cnn_8 = my_new_axis(input_image_for_cnn_8)

for i in range(len(file_list0_8)):

    img = pickle.load(open(file_list0_8[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_8 = np.append(input_image_for_cnn_8, img, axis = 0)


input_image_for_cnn_8 = input_image_for_cnn_8[1:,:,:,:]

for j in range(len(file_list1_8)):

    img = pickle.load(open(file_list1_8[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_8 = np.append(input_image_for_cnn_8, img, axis = 0)


print(input_image_for_cnn_8.shape)

# 9번째 CNN에 들어갈 Input임


input_image_for_cnn_9 = pickle.load(open(file_list0_9[0],'rb'))
input_image_for_cnn_9 = my_new_axis(input_image_for_cnn_9)

for i in range(len(file_list0_9)):

    img = pickle.load(open(file_list0_9[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_9 = np.append(input_image_for_cnn_9, img, axis = 0)


input_image_for_cnn_9 = input_image_for_cnn_9[1:,:,:,:]

for j in range(len(file_list1_9)):

    img = pickle.load(open(file_list1_9[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_9 = np.append(input_image_for_cnn_9, img, axis = 0)


print(input_image_for_cnn_9.shape)

# 10번째 CNN에 들어갈 Input임


input_image_for_cnn_10 = pickle.load(open(file_list0_10[0],'rb'))
input_image_for_cnn_10 = my_new_axis(input_image_for_cnn_10)

for i in range(len(file_list0_10)):

    img = pickle.load(open(file_list0_10[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_10 = np.append(input_image_for_cnn_10, img, axis = 0)


input_image_for_cnn_10 = input_image_for_cnn_10[1:,:,:,:]

for j in range(len(file_list1_10)):

    img = pickle.load(open(file_list1_10[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_10 = np.append(input_image_for_cnn_10, img, axis = 0)


print(input_image_for_cnn_10.shape)

# 11번째 CNN에 들어갈 Input임


input_image_for_cnn_11 = pickle.load(open(file_list0_11[0],'rb'))
input_image_for_cnn_11 = my_new_axis(input_image_for_cnn_11)

for i in range(len(file_list0_11)):

    img = pickle.load(open(file_list0_11[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_11 = np.append(input_image_for_cnn_11, img, axis = 0)


input_image_for_cnn_11 = input_image_for_cnn_11[1:,:,:,:]

for j in range(len(file_list1_11)):

    img = pickle.load(open(file_list1_11[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_11 = np.append(input_image_for_cnn_11, img, axis = 0)


print(input_image_for_cnn_11.shape)

# 12번째 CNN에 들어갈 Input임


input_image_for_cnn_12 = pickle.load(open(file_list0_12[0],'rb'))
input_image_for_cnn_12 = my_new_axis(input_image_for_cnn_12)

for i in range(len(file_list0_12)):

    img = pickle.load(open(file_list0_12[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_12 = np.append(input_image_for_cnn_12, img, axis = 0)


input_image_for_cnn_12 = input_image_for_cnn_12[1:,:,:,:]

for j in range(len(file_list1_12)):

    img = pickle.load(open(file_list1_12[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_12 = np.append(input_image_for_cnn_12, img, axis = 0)


print(input_image_for_cnn_12.shape)

# 13번째 CNN에 들어갈 Input임


input_image_for_cnn_13 = pickle.load(open(file_list0_13[0],'rb'))
input_image_for_cnn_13 = my_new_axis(input_image_for_cnn_13)

for i in range(len(file_list0_13)):

    img = pickle.load(open(file_list0_13[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_13 = np.append(input_image_for_cnn_13, img, axis = 0)


input_image_for_cnn_13 = input_image_for_cnn_13[1:,:,:,:]

for j in range(len(file_list1_13)):

    img = pickle.load(open(file_list1_13[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_13 = np.append(input_image_for_cnn_13, img, axis = 0)


print(input_image_for_cnn_13.shape)

# 14번째 CNN에 들어갈 Input임


input_image_for_cnn_14 = pickle.load(open(file_list0_14[0],'rb'))
input_image_for_cnn_14 = my_new_axis(input_image_for_cnn_14)

for i in range(len(file_list0_14)):

    img = pickle.load(open(file_list0_14[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_14 = np.append(input_image_for_cnn_14, img, axis = 0)


input_image_for_cnn_14 = input_image_for_cnn_14[1:,:,:,:]

for j in range(len(file_list1_14)):

    img = pickle.load(open(file_list1_14[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_14 = np.append(input_image_for_cnn_14, img, axis = 0)


print(input_image_for_cnn_14.shape)

# 15번째 CNN에 들어갈 Input임


input_image_for_cnn_15 = pickle.load(open(file_list0_15[0],'rb'))
input_image_for_cnn_15 = my_new_axis(input_image_for_cnn_15)

for i in range(len(file_list0_15)):

    img = pickle.load(open(file_list0_15[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_15 = np.append(input_image_for_cnn_15, img, axis = 0)


input_image_for_cnn_15 = input_image_for_cnn_15[1:,:,:,:]

for j in range(len(file_list1_15)):

    img = pickle.load(open(file_list1_15[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_15 = np.append(input_image_for_cnn_15, img, axis = 0)


print(input_image_for_cnn_15.shape)

# 16번째 CNN에 들어갈 Input임


input_image_for_cnn_16 = pickle.load(open(file_list0_16[0],'rb'))
input_image_for_cnn_16 = my_new_axis(input_image_for_cnn_16)

for i in range(len(file_list0_16)):

    img = pickle.load(open(file_list0_16[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_16 = np.append(input_image_for_cnn_16, img, axis = 0)


input_image_for_cnn_16 = input_image_for_cnn_16[1:,:,:,:]

for j in range(len(file_list1_16)):

    img = pickle.load(open(file_list1_16[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_16 = np.append(input_image_for_cnn_16, img, axis = 0)


print(input_image_for_cnn_16.shape)

# 17번째 CNN에 들어갈 Input임


input_image_for_cnn_17 = pickle.load(open(file_list0_17[0],'rb'))
input_image_for_cnn_17 = my_new_axis(input_image_for_cnn_17)

for i in range(len(file_list0_17)):

    img = pickle.load(open(file_list0_17[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_17 = np.append(input_image_for_cnn_17, img, axis = 0)


input_image_for_cnn_17 = input_image_for_cnn_17[1:,:,:,:]

for j in range(len(file_list1_17)):

    img = pickle.load(open(file_list1_17[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_17 = np.append(input_image_for_cnn_17, img, axis = 0)


print(input_image_for_cnn_17.shape)

# 18번째 CNN에 들어갈 Input임


input_image_for_cnn_18 = pickle.load(open(file_list0_18[0],'rb'))
input_image_for_cnn_18 = my_new_axis(input_image_for_cnn_18)

for i in range(len(file_list0_18)):

    img = pickle.load(open(file_list0_18[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_18 = np.append(input_image_for_cnn_18, img, axis = 0)


input_image_for_cnn_18 = input_image_for_cnn_18[1:,:,:,:]

for j in range(len(file_list1_18)):

    img = pickle.load(open(file_list1_18[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_18 = np.append(input_image_for_cnn_18, img, axis = 0)


print(input_image_for_cnn_18.shape)

# 19번째 CNN에 들어갈 Input임


input_image_for_cnn_19 = pickle.load(open(file_list0_19[0],'rb'))
input_image_for_cnn_19 = my_new_axis(input_image_for_cnn_19)

for i in range(len(file_list0_19)):

    img = pickle.load(open(file_list0_19[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_19 = np.append(input_image_for_cnn_19, img, axis = 0)


input_image_for_cnn_19 = input_image_for_cnn_19[1:,:,:,:]

for j in range(len(file_list1_19)):

    img = pickle.load(open(file_list1_19[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_19 = np.append(input_image_for_cnn_19, img, axis = 0)


print(input_image_for_cnn_19.shape)

# 20번째 CNN에 들어갈 Input임


input_image_for_cnn_20 = pickle.load(open(file_list0_20[0],'rb'))
input_image_for_cnn_20 = my_new_axis(input_image_for_cnn_20)

for i in range(len(file_list0_20)):

    img = pickle.load(open(file_list0_20[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_20 = np.append(input_image_for_cnn_20, img, axis = 0)


input_image_for_cnn_20 = input_image_for_cnn_20[1:,:,:,:]

for j in range(len(file_list1_20)):

    img = pickle.load(open(file_list1_20[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_20 = np.append(input_image_for_cnn_20, img, axis = 0)


print(input_image_for_cnn_20.shape)

# 21번째 CNN에 들어갈 Input임


input_image_for_cnn_21 = pickle.load(open(file_list0_21[0],'rb'))
input_image_for_cnn_21 = my_new_axis(input_image_for_cnn_21)

for i in range(len(file_list0_21)):

    img = pickle.load(open(file_list0_21[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_21 = np.append(input_image_for_cnn_21, img, axis = 0)


input_image_for_cnn_21 = input_image_for_cnn_21[1:,:,:,:]

for j in range(len(file_list1_21)):

    img = pickle.load(open(file_list1_21[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_21 = np.append(input_image_for_cnn_21, img, axis = 0)


print(input_image_for_cnn_21.shape)

# 22번째 CNN에 들어갈 Input임


input_image_for_cnn_22 = pickle.load(open(file_list0_22[0],'rb'))
input_image_for_cnn_22 = my_new_axis(input_image_for_cnn_22)

for i in range(len(file_list0_22)):

    img = pickle.load(open(file_list0_22[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_22 = np.append(input_image_for_cnn_22, img, axis = 0)


input_image_for_cnn_22 = input_image_for_cnn_22[1:,:,:,:]

for j in range(len(file_list1_22)):

    img = pickle.load(open(file_list1_22[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_22 = np.append(input_image_for_cnn_22, img, axis = 0)


print(input_image_for_cnn_22.shape)

# 23번째 CNN에 들어갈 Input임


input_image_for_cnn_23 = pickle.load(open(file_list0_23[0],'rb'))
input_image_for_cnn_23 = my_new_axis(input_image_for_cnn_23)

for i in range(len(file_list0_23)):

    img = pickle.load(open(file_list0_23[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_23 = np.append(input_image_for_cnn_23, img, axis = 0)


input_image_for_cnn_23 = input_image_for_cnn_23[1:,:,:,:]

for j in range(len(file_list1_23)):

    img = pickle.load(open(file_list1_23[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_23 = np.append(input_image_for_cnn_23, img, axis = 0)


print(input_image_for_cnn_23.shape)

# 24번째 CNN에 들어갈 Input임


input_image_for_cnn_24 = pickle.load(open(file_list0_24[0],'rb'))
input_image_for_cnn_24 = my_new_axis(input_image_for_cnn_24)

for i in range(len(file_list0_24)):

    img = pickle.load(open(file_list0_24[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_24 = np.append(input_image_for_cnn_24, img, axis = 0)


input_image_for_cnn_24 = input_image_for_cnn_24[1:,:,:,:]

for j in range(len(file_list1_24)):

    img = pickle.load(open(file_list1_24[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_24 = np.append(input_image_for_cnn_24, img, axis = 0)


print(input_image_for_cnn_24.shape)

# 25번째 CNN에 들어갈 Input임


input_image_for_cnn_25 = pickle.load(open(file_list0_25[0],'rb'))
input_image_for_cnn_25 = my_new_axis(input_image_for_cnn_25)

for i in range(len(file_list0_25)):

    img = pickle.load(open(file_list0_25[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_25 = np.append(input_image_for_cnn_25, img, axis = 0)


input_image_for_cnn_25 = input_image_for_cnn_25[1:,:,:,:]

for j in range(len(file_list1_25)):

    img = pickle.load(open(file_list1_25[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_25 = np.append(input_image_for_cnn_25, img, axis = 0)


print(input_image_for_cnn_25.shape)

# 26번째 CNN에 들어갈 Input임


input_image_for_cnn_26 = pickle.load(open(file_list0_26[0],'rb'))
input_image_for_cnn_26 = my_new_axis(input_image_for_cnn_26)

for i in range(len(file_list0_26)):

    img = pickle.load(open(file_list0_26[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_26 = np.append(input_image_for_cnn_26, img, axis = 0)


input_image_for_cnn_26 = input_image_for_cnn_26[1:,:,:,:]

for j in range(len(file_list1_26)):

    img = pickle.load(open(file_list1_26[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_26 = np.append(input_image_for_cnn_26, img, axis = 0)


print(input_image_for_cnn_26.shape)

# 27번째 CNN에 들어갈 Input임


input_image_for_cnn_27 = pickle.load(open(file_list0_27[0],'rb'))
input_image_for_cnn_27 = my_new_axis(input_image_for_cnn_27)

for i in range(len(file_list0_27)):

    img = pickle.load(open(file_list0_27[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_27 = np.append(input_image_for_cnn_27, img, axis = 0)


input_image_for_cnn_27 = input_image_for_cnn_27[1:,:,:,:]

for j in range(len(file_list1_27)):

    img = pickle.load(open(file_list1_27[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_27 = np.append(input_image_for_cnn_27, img, axis = 0)


print(input_image_for_cnn_27.shape)

# 28번째 CNN에 들어갈 Input임


input_image_for_cnn_28 = pickle.load(open(file_list0_28[0],'rb'))
input_image_for_cnn_28 = my_new_axis(input_image_for_cnn_28)

for i in range(len(file_list0_28)):

    img = pickle.load(open(file_list0_28[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_28 = np.append(input_image_for_cnn_28, img, axis = 0)


input_image_for_cnn_28 = input_image_for_cnn_28[1:,:,:,:]

for j in range(len(file_list1_28)):

    img = pickle.load(open(file_list1_28[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_28 = np.append(input_image_for_cnn_28, img, axis = 0)


print(input_image_for_cnn_28.shape)

# 29번째 CNN에 들어갈 Input임


input_image_for_cnn_29 = pickle.load(open(file_list0_29[0],'rb'))
input_image_for_cnn_29 = my_new_axis(input_image_for_cnn_29)

for i in range(len(file_list0_29)):

    img = pickle.load(open(file_list0_29[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_29 = np.append(input_image_for_cnn_29, img, axis = 0)


input_image_for_cnn_29 = input_image_for_cnn_29[1:,:,:,:]

for j in range(len(file_list1_29)):

    img = pickle.load(open(file_list1_29[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_29 = np.append(input_image_for_cnn_29, img, axis = 0)


print(input_image_for_cnn_29.shape)

# 30번째 CNN에 들어갈 Input임


input_image_for_cnn_30 = pickle.load(open(file_list0_30[0],'rb'))
input_image_for_cnn_30 = my_new_axis(input_image_for_cnn_30)

for i in range(len(file_list0_30)):

    img = pickle.load(open(file_list0_30[i],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_30 = np.append(input_image_for_cnn_30, img, axis = 0)


input_image_for_cnn_30 = input_image_for_cnn_30[1:,:,:,:]

for j in range(len(file_list1_30)):

    img = pickle.load(open(file_list1_30[j],'rb'))
    img = img/255.0
    img = my_new_axis(img)
    input_image_for_cnn_30 = np.append(input_image_for_cnn_30, img, axis = 0)


print(input_image_for_cnn_30.shape)















zeros = np.zeros(138)
ones = np.ones(39)
y_label = np.hstack((zeros,ones))
y_label = y_label.reshape(len(y_label),1)
y_label_cat = to_categorical(y_label,2)


# In[ ]:

for iii in range(3):
    model.fit([input_image_for_cnn_1,input_image_for_cnn_2,
    input_image_for_cnn_3,input_image_for_cnn_4,input_image_for_cnn_5,input_image_for_cnn_6,
    input_image_for_cnn_7,input_image_for_cnn_8,input_image_for_cnn_9,input_image_for_cnn_10,
    input_image_for_cnn_11,input_image_for_cnn_12,input_image_for_cnn_13,input_image_for_cnn_14,
    input_image_for_cnn_15,input_image_for_cnn_16,input_image_for_cnn_17,input_image_for_cnn_18,
    input_image_for_cnn_19,input_image_for_cnn_20,input_image_for_cnn_21,input_image_for_cnn_22,
    input_image_for_cnn_23,input_image_for_cnn_24,input_image_for_cnn_25,input_image_for_cnn_26,
    input_image_for_cnn_27,input_image_for_cnn_28,input_image_for_cnn_29,input_image_for_cnn_30], y_label_cat, epochs=10, batch_size=8, validation_split=0.4)

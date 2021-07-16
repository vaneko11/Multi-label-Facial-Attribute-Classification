import numpy as np
import os
import pandas as pd
import keras
from keras.preprocessing import image as img
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import metrics


#y_read
y = pd.read_csv("atr_10.txt", sep=" ",header=None)
y=y.T.loc[:999]
y=y.values

#img_read
files = sorted(os.listdir('Img'))

x=[]
for f in files:
	image = img.load_img('Img/'+f, grayscale=False, color_mode='rgb')
	im_ar = img.img_to_array(image)
	x.append(im_ar)

X = np.array(x)	

print(y.shape)
print(X.shape)

#data_procesing

train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size = 0.3, random_state = 42)

train_X=train_X.astype('float32')
test_X=test_X.astype('float32')
 
train_X=train_X/255.0
test_X=test_X/255.0

 
#num_classes=test_Y.shape[1]

#network
batch_size = 16
epochs = 32

model = Sequential()

#### Input Layer ####
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same',
                 activation='relu', input_shape=( 218,178, 3)))

#### Convolutional Layers ####
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))  # Pooling

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(Conv2D(512, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

#### Fully-Connected Layer ####
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(),metrics=['Recall'])

print(model.summary()) 

#train
fashion_train = model.fit(train_X, train_Y, batch_size=batch_size,epochs=epochs,verbose=1)

#test
test_eval = model.evaluate(test_X, test_Y, verbose=0)
print('Test loss:', test_eval[0])
print('Test recall:', test_eval[1])

with open('sample_cnn_prec.txt', "w") as fl:
		fl.write('Test loss:' + str(test_eval[0]) +'\n' + 'Test recall:' + str(test_eval[1]) + '\n')

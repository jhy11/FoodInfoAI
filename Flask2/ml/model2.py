import scipy.io
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


from keras.models import load_model
from keras.preprocessing import image
from keras import optimizers

import numpy as np
from os import listdir
from os.path import isfile, join

img_width = 100
img_height = 100

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
train_samples = 63*10
validation_samples = 21*10
epochs = 5
batch_size = 15

# Check for TensorFlow or Thieno
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
model = Sequential()
# Conv2D : Two dimenstional convulational model.
# 32 : Input for next layer
# (3,3) convulonational windows size
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # Output convert into one dimension layer and will go to Dense layer
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()


model.compile(loss='categorical_crossentropy', 
              optimizer=optimizers.Adam(lr=.0004),
              metrics=['accuracy'])

# Data Augmentation is a method of artificially creating a new dataset for training from 
# the existing training dataset to improve the performance of deep learning neural network 
# with the amount of data available. It is a form of regularization which makes our model generalize better 
# than before.

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling, avoiding having same training and validation data.
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

print(train_generator.class_indices)

imgs, labels = next(train_generator)

from skimage import io

def imshow(image_RGB):
  io.imshow(image_RGB)
  io.show()

import matplotlib.pyplot as plt

image_batch,label_batch = train_generator.next()

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


#In fit_generator(), you don't pass the x and y directly, instead they come from a generator.
history = model.fit(
    train_generator,
    steps_per_epoch=train_samples // batch_size,
    
    
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size)

import matplotlib.pyplot as plt

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

## Now Predict
predict_dir_path='data/test/'
onlyfiles = [f for f in listdir(predict_dir_path) if isfile(join(predict_dir_path, f))]
print(onlyfiles)

# predicting images
from keras.preprocessing import image
import webbrowser

chokchok_counter = 0 
coffee_counter  = 0
enaak_counter  = 0
ggobuk_counter = 0 
hotdog_counter  = 0
hush_counter  = 0
oreo_counter  = 0
pepero_counter  = 0
poka_counter = 0 
twix_counter  = 0

for file in onlyfiles:
    img = image.load_img(predict_dir_path+file, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    classes = classes
    
    if classes == 0:
        print(file + ": " + 'chockchok')
        chokchok_counter += 1
        #webbrowser.open("http://www.daum.net")
    elif classes==1:
        print(file + ": " + 'coffee')
        coffee_counter += 1
       # webbrowser.open("http://www.naver.com")
    elif classes== 2:
        print(file + ": " + 'enaak')
        enaak_counter += 1
        #webbrowser.open("http://www.google.com")
    elif classes==3:
        print(file + ": " + 'ggobuk')
        ggobuk_counter += 1
       # webbrowser.open("http://www.naver.com")
    elif classes== 4:
        print(file + ": " + 'hotdog')
        hotdog_counter += 1
        #webbrowser.open("http://www.google.com")
    elif classes==5:
        print(file + ": " + 'hush')
        hush_counter += 1
       # webbrowser.open("http://www.naver.com")
    elif classes==6:
        print(file + ": " + 'oreo')
        oreo_counter += 1
       # webbrowser.open("http://www.naver.com")
    elif classes== 7:
        print(file + ": " + 'pepero')
        pepero_counter += 1
        #webbrowser.open("http://www.google.com")
    elif classes==8:
        print(file + ": " + 'poka')
        poka_counter += 1
       # webbrowser.open("http://www.naver.com")
    elif classes== 9:
        print(file + ": " + 'twix')
        twix_counter += 1
        #webbrowser.open("http://www.google.com")
print("Total chokchok :",chokchok_counter)
print("Total coffee :",coffee_counter)
print("Total enaak :",enaak_counter)
print("Total ggobuk :",ggobuk_counter)
print("Total hotdog :",hotdog_counter)
print("Total hush :",hush_counter)
print("Total oreo :",oreo_counter)
print("Total pepero :",pepero_counter)
print("Total poka :",poka_counter)
print("Total twix :",twix_counter)

model.save("model2.h5")


# Google 주소 숫자 인식 모델 생성

# 로드 mat 파일
#train_data = scipy.io.loadmat('extra_32x32.mat')

# 학습 데이터, 훈련 데이터
#X = train_data['X']
#y = train_data['y']

# 매트릭스 1D 변환
#X = X.reshape(X.shape[0] * X.shape[1] * X.shape[2], X.shape[3]).T
#y = y.reshape(y.shape[0], )

# 셔플(섞기)
#X, y = shuffle(X, y, random_state=42)

# 학습 훈련 데이터 분리
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# 랜덤 포레스트 객체 생성 및 학습
#clf = RandomForestClassifier()
#clf.fit(X_train, y_train)

# 모델 저장
#joblib.dump(clf, '../model/model.pkl')

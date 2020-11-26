import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
from scipy import misc
import imageio
import keras

import matplotlib.pyplot as plt

import webbrowser
import urllib.request


# 모델 로드
    # ml/model.py 선 실행 후 생성
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
import os
from os.path import isfile, join
from skimage import io
from keras.preprocessing import image
app = Flask(__name__)


# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


# 데이터 예측 처리
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':

        # 업로드 파일 처리 분기
        file = request.files['image']
        if not file: return render_template('index.html', label="No Files")
        file.save('./temp/'+file.filename)


        ## Now Predict

        predict_dir_path='./temp/'
        onlyfiles = [f for f in listdir(predict_dir_path) if isfile(join(predict_dir_path, f))]

        for file in onlyfiles:
            img = image.load_img(predict_dir_path+file, target_size=(img_width, img_height))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            
            images = np.vstack([x])
            classes = model.predict_classes(images, batch_size=10)
            classes = classes
            os.remove('./temp/'+file)
            if classes == 0:
                print(file + ": " + 'chockchok')
                return render_template('chokchok.html')
            elif classes==1:
                print(file + ": " + 'coffee')
                return render_template('coffee.html')
            elif classes== 2:
                print(file + ": " + 'enaak')
                return render_template('enaak.html')
            elif classes==3:
                print(file + ": " + 'ggobuk')
                return render_template('ggobuk.html')
            elif classes== 4:
                print(file + ": " + 'hotdog')
                return render_template('hotdog.html')
            elif classes==5:
                print(file + ": " + 'hush')
                return render_template('hershey.html')
            elif classes==6:
                print(file + ": " + 'oreo')
                return render_template('oreo.html')
            elif classes== 7:
                print(file + ": " + 'pepero')
                return render_template('pepero.html')
            elif classes==8:
                print(file + ": " + 'poka')
                return render_template('poca.html')
            elif classes== 9:
                print(file + ": " + 'twix')
                return render_template('twix.html')


if __name__ == '__main__':
    

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
    model.load_weights("C:/Users/shims/FoodComponent/CNN_Model.h5")

    # Data Augmentation is a method of artificially creating a new dataset for training from 
    # the existing training dataset to improve the performance of deep learning neural network 
    # with the amount of data available. It is a form of regularization which makes our model generalize better 
    # than before.

    
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=True)

# import openai
# import os
# import json
# from base64 import b64decode
#
# Api_cey = ""
# openai.api_key = Api_cey
#
# prompt = "capital letter h in black on white background"
# response = openai.Image.create(
#     prompt=prompt,
#     n=1,
#     size='256x256',
#     response_format='b64_json'
# )
#
# with open('data.json', 'w') as file:
#     json.dump(response, file, indent=4, ensure_ascii=False)
#
# image_data = b64decode(response['data'][0]['b64_json'])
# file_name = '_'.join(prompt.split(' '))
#
# with open(f'{file_name}.png', 'wb') as file:
#     file.write(image_data)
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import cv2
import matplotlib.pyplot as plt
import idx2numpy
from keras.preprocessing.image import ImageDataGenerator
import sys # Для sys.exit()
#
pathToData = ''
num_classes = 26 #10
img_cols = img_rows = 28
# Флаг вывода изображений букв после загрузки данных
showPics = False # True False
# Флаг вывода изображений букв из тестового или обучающего набора (вывод после загрузки данных)
showFromTest = True
X_train_shape_0 = 124800 #60000
X_test_shape_0 = 20800 #10000
#
# Загрузка EMNIST
def load_data():
    # Загрузка данных
    imagesTrain, labelsTrain, imagesTest, labelsTest = loadBinData(pathToData)

    #
    X_train = np.asarray(imagesTrain)
    X_train = X_train[16:]
    y_train = np.asarray(labelsTrain)
    y_train = y_train[8:]
    X_test = np.asarray(imagesTest)
    X_test = X_test[16:]
    y_test = np.asarray(labelsTest)
    y_test = y_test[8:]
    y_train -= 1
    y_test -= 1
    y_test_0 = y_test # Для predict в checkModel (43)
    X_train = X_train.reshape(X_train_shape_0, img_rows, img_cols, 1).transpose(0, 2, 1, 3)
    X_test = X_test.reshape(X_test_shape_0, img_rows, img_cols, 1).transpose(0, 2, 1, 3)
    if showPics:
        if showFromTest:
            print('Показываем примеры тестовых данных')
        else:
            print('Показываем примеры обучающих данных')
        # Выводим 16 изображений обучающего или тестового набора
        names = makeNames()
        for i in range(64):
            plt.subplot(4, 16, i + 1)
            ind = y_test[i] if showFromTest else y_train[i]
            ind = (ind % 26)# + 1
            img = X_test[i][0:img_rows, 0:img_rows, 0] if showFromTest else X_train[i][0:img_rows, 0:img_rows, 0]
            plt.imshow(img, cmap = plt.get_cmap('gray'))
            plt.title(names[ind])
            plt.axis('off')
        plt.subplots_adjust(hspace = 0.1) # wspace
        plt.show()
        sys.exit()
    return X_train, y_train, X_test, y_test
#
# def loadBinData(pathToData):
#     print('Загрузка данных из двоичных файлов')
#     with open('emnist-letters-train-images-idx3-ubyte', 'rb') as read_binary:
#         data = np.fromfile(read_binary, dtype = np.uint8)
#     with open('emnist-letters-train-labels-idx1-ubyte', 'rb') as read_binary:
#         labels = np.fromfile(read_binary, dtype = np.uint8)
#     with open('emnist-letters-test-images-idx3-ubyte', 'rb') as read_binary:
#         data2 = np.fromfile(read_binary, dtype = np.uint8)
#     with open('emnist-letters-test-labels-idx1-ubyte', 'rb') as read_binary:
#         labels2 = np.fromfile(read_binary, dtype = np.uint8)
#     return data, labels, data2, labels2
def loadBinData(pathToData):
    print('Загрузка данных из двоичных файлов')
    with open('emnist-letters-train-images-idx3-ubyte', 'rb') as read_binary:
        data = np.fromfile(read_binary, dtype = np.uint8)
    with open('emnist-letters-train-labels-idx1-ubyte', 'rb') as read_binary:
        labels = np.fromfile(read_binary, dtype = np.uint8)
    with open('emnist-letters-test-images-idx3-ubyte', 'rb') as read_binary:
        data2 = np.fromfile(read_binary, dtype = np.uint8)
    with open('emnist-letters-test-labels-idx1-ubyte', 'rb') as read_binary:
        labels2 = np.fromfile(read_binary, dtype = np.uint8)
    return data, labels, data2, labels2
#
def makeNames():
    names = []
    for i in range(26):
        # if i > 25 and i < 32:
            # continue
        names.append(chr(65 + i)) #48 ['A', 'B', 'C', ..., 'Z']
    print(names)
    return names
#
# Главная программа
if __name__ == '__main__':
    # num_classes = 10
    # input_shape = (28, 28, 1)
    #
    # # Load the data and split it between train and test sets
    # # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # # print("X_train shape", x_train.shape)
    # # print(type(x_train))
    # x_train = idx2numpy.convert_from_file('train-images.idx3-ubyte')
    # # y_train = np.fromfile('train-labels.idx1-ubyte')
    # # x_test = np.fromfile('t10k-images.idx3-ubyte')
    # # y_test = np.fromfile('t10k-labels.idx1-ubyte')
    # print(x_train)
    # # with open('train-images.idx3-ubyte', 'rb') as read_binary:
    # #     data = np.fromfile(read_binary, dtype=np.uint8)
    # # print(len(data))
    # # # Scale images to the [0, 1] range
    # # x_train = x_train.astype("float32") / 255
    # # x_test = x_test.astype("float32") / 255
    # # # Make sure images have shape (28, 28, 1)
    # # x_train = np.expand_dims(x_train, -1)
    # # x_test = np.expand_dims(x_test, -1)
    # # print("x_train shape:", x_train.shape)
    # # print(x_train.shape[0], "train samples")
    # # print(x_test.shape[0], "test samples")
    # #
    # # # convert class vectors to binary class matrices
    # # y_train = keras.utils.to_categorical(y_train, num_classes)
    # # y_test = keras.utils.to_categorical(y_test, num_classes)
    # #
    # # """
    # # ## Build the model
    # # """
    #
    # # model = keras.Sequential(
    # #     [
    # #         keras.Input(shape=input_shape),
    # #         layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    # #         layers.MaxPooling2D(pool_size=(2, 2)),
    # #         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    # #         layers.MaxPooling2D(pool_size=(2, 2)),
    # #         layers.Flatten(),
    # #         layers.Dropout(0.5),
    # #         layers.Dense(num_classes, activation="softmax"),
    # #     ]
    # # )
    # #
    # # model.summary()
    # #
    # # """
    # # ## Train the model
    # # """
    # #
    # # batch_size = 128
    # # epochs = 15
    # #
    # # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # #
    # # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    # #
    # # """
    # # ## Evaluate the trained model
    # # """
    # #
    # # score = model.evaluate(x_test, y_test, verbose=0)
    # # print("Test loss:", score[0])
    # # print("Test accuracy:", score[1])






    # Загрузка данных и формирование обучающих и тестовых выборок
    X_train, y_train, X_test, y_test = load_data()
    # Все параметры имеют заданные по умолчанию значения
    datagen = ImageDataGenerator()
    print('Настройка генератора...')
    datagen.fit(X_train)
    print('Получаем сгенерированные образы и показываем 15 первых экземпляров заданной буквы, например, A (a)...')
    X_y = datagen.flow(X_test, y_test, batch_size = 1) # batch_size = 32
    names = makeNames()
    print(type(X_y)) # class 'keras_preprocessing.image.NumpyArrayIterator'
    print(len(X_y)) # 20800
    print(names[X_y[0][1][0]])
    str1 = 'ABCEIKMOVY'
    numImage = 0
    for jj in str1:
        # print(jj)
        letToShow = jj
        k = 0
        # for i in range(len(X_y)):
        #     ind = X_y[i][1][0]
        #     print(ind)
        image1 = 0
        for i in range(len(X_y)):
            ind = X_y[i][1][0]
            # print(ind)
            let = names[(ind % 26)] #+1
            if let == letToShow:
                # numImage += 1
                k += 1
                # plt.subplot(2, 8, k + 1)
                img = X_y[i][0].astype('uint8')
                img = img.reshape(img_rows, img_cols)
                for i in img:
                    for j in range(0, 28):
                        if i[j] > 100:
                            i[j] = 0
                        else:
                            i[j] = 255

                # print(img)
                cv2.imshow('28x28 white', img)
                #if k > 100:

                print(cv2.imwrite(f'{jj}image/test{k}({jj}).png', img))
                # print(cv2.imwrite(f'Aimage/test1(A).png', img))
                # if k == 0:
                    # print(img)
                # plt.imshow(img, cmap = plt.get_cmap('gray'))
                # plt.title(let)
                # plt.axis('off')
                if k == 50:
                    # numImage += k
                    break
# plt.subplots_adjust(hspace = 0.1) # wspace
# plt.show()
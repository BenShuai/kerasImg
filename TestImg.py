from PIL import Image

# im=Image.open("D:/timg.jpg") #打开的图像对象
# cur_pixel = im.getpixel((10, 10)) # 获得图像的rgb值
# print(cur_pixel)
#
# imL=im.convert('L') # 灰度图
# # imL.save("D:/微信图片_20180911142205HL.jpg")
# imL.show()
# cur_pixel = imL.getpixel((10, 10)) # 获得图像的rgb值
# print(cur_pixel)


import numpy as np
import os

import keras
from keras.models import Sequential
from keras.layers import Dense

'''
返回某个目录下面的图像的像素信息和标签
'''
def getImgData(path):
    colors=[]
    lables=[]
    files = os.listdir(path) # 某个目录下文件的列表
    for j in files:
        try:
            im = Image.open(path+j) # 读取原图
            w=im.width # 原图的宽
            h=im.height # 原图的高

            for x in range(w-3):
                for y in range(h-3):
                    box = (x,y,x+3,y+3) #创建一个选区范围【左、上、右、下】
                    out= im.crop(box)  #截取一个3*3的图像

                    imL = out.convert('L') # 灰度图
                    colors.append(np.array(imL))

                    cur_pixel = im.getpixel((x+1, y+1)) # 获得图像的rgb值
                    if type(cur_pixel) != type((0,0,0)): # 判断类型，原图RGB是一个有3个值的元祖类型
                        break

                    lablesColorVar="" # 标签值，R+G+B 组合的值,某个色道的值不足3位就补0  如 原始RGB为 （10,100,30） = 010 100 030 = 010100030
                    if len(str(cur_pixel[0]))==3:
                        lablesColorVar+=str(cur_pixel[0])
                    elif len(str(cur_pixel[0]))==2:
                        lablesColorVar+="0"+str(cur_pixel[0])
                    elif len(str(cur_pixel[0]))==1:
                        lablesColorVar+="00"+str(cur_pixel[0])

                    if len(str(cur_pixel[1]))==3:
                        lablesColorVar+=str(cur_pixel[1])
                    elif len(str(cur_pixel[1]))==2:
                        lablesColorVar+="0"+str(cur_pixel[1])
                    elif len(str(cur_pixel[1]))==1:
                        lablesColorVar+="00"+str(cur_pixel[1])

                    if len(str(cur_pixel[2]))==3:
                        lablesColorVar+=str(cur_pixel[2])
                    elif len(str(cur_pixel[2]))==2:
                        lablesColorVar+="0"+str(cur_pixel[2])
                    elif len(str(cur_pixel[2]))==1:
                        lablesColorVar+="00"+str(cur_pixel[2])

                    lables.append(int(lablesColorVar))

                    lablesColorVar=None
                    imL=None
            im=None
            print(j,"加载完毕")
        except:
            pass

    # print(colors)
    # print(lables)

    colors = np.array(colors)
    lables = np.array(lables)
    return (colors,lables)


def createDataSet():
    colors=[]
    lables=[]
    path = "D:/AI/gl/" # 训练集素材
    (colors,lables) = getImgData(path)

    testColors=[]
    testLables=[]
    path2 = "D:/AI/gl2/" # 测试集素材
    (testColors,testLables) = getImgData(path2)
    return (colors,lables,testColors,testLables)

if __name__=='__main__':
    num_classes = 255255255
    (colors,lables,testColors,testLables) = createDataSet()

    colors = colors.reshape(colors.shape[0], 9).astype('float32')
    colors /= 255

    testColors = testColors.reshape(testColors.shape[0], 9).astype('float32')
    testColors /= 255

    lables = keras.utils.to_categorical(lables, num_classes)
    testLables = keras.utils.to_categorical(testLables, num_classes)

    print(colors.shape,"  ",lables.shape)
    print(testColors.shape,"  ",testLables.shape)

    model = Sequential()
    model.add(Dense(num_classes*3, activation='relu',input_shape=(9,)))
    model.add(Dense(num_classes*2, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    model.fit(colors, lables, batch_size=128, epochs=5, verbose=1, validation_data=(testColors, testLables))
    score = model.evaluate(testColors, testLables, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]) # 准确度

    model.save('Test_model_weights.h5') # 保存训练模型


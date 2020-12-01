import glob, os, keras, tensorflow
import numpy as np
import matplotlib as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from keras.models import Sequential
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.filedialog

def get_data():
    # 训练所需图片的路径
    train_dir = './garbage_classify/train_data'
    # 训练集数据
    train_datagen = ImageDataGenerator(
        rescale=1. / 225, shear_range=0.1, zoom_range=0.1,
        width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,
        vertical_flip=True, validation_split=0.1)
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(300, 300), batch_size=16,
        class_mode='categorical', subset='training', seed=0)

    # 测试集数据
    test_datagen = ImageDataGenerator(
        rescale=1. / 255, validation_split=0.1)
    validation_generator = test_datagen.flow_from_directory(
        train_dir, target_size=(300, 300), batch_size=16,
        class_mode='categorical', subset='validation', seed=0)

    # 获取文件夹名字作为标签的名字
    labels = train_generator.class_indices
    labels = dict((v,k) for k,v in labels.items())
    print(labels)
    return train_generator, validation_generator, labels

# 创建模型
def model_train_function(train_generator):
    model = Sequential([
        Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(300, 300, 3)),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(4, activation='softmax')
    ])
    #开始训练模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit_generator(train_generator, epochs=100, steps_per_epoch=2276//32,validation_data=validation_generator,
                        validation_steps=251//32)

    # 保存模型
    model.save('model.h5')

def test_function():
    # 获取测试数据和标签
    test_x, test_y = validation_generator.__getitem__(1)
    print(type(test_x), test_x.shape)
    # 调用模型
    new_model = keras.models.load_model('model.h5')
    loss, acc = new_model.evaluate(test_x, test_y)
    print(loss, acc)
    preds = new_model.predict(test_x)
    # 显示预测结果
    for i in range(len(preds)):
        print(labels[np.argmax(preds[i])])


train_generator, validation_generator, labels = get_data()

# model_train_function(train_generator)

# 创建窗口
i = 'img_1.jpg'
TK = tk.Tk()
TK.title('第四组-专业骗鱼')
TK.geometry('800x500')


def start(load):
    im = Image.open(load)
    x = im.resize((300, 300), Image.ANTIALIAS)
    x = np.array(x)
    x = np.reshape(x, (1, 300, 300, 3))
    print(type(x), x.shape)
    new_model = keras.models.load_model('model.h5')
    preds = new_model.predict(x)
    preds_result = labels[np.argmax(preds)]
    print(preds_result)
    im = Image.open(load)
    im = im.resize((400, 400))
    global img
    img = ImageTk.PhotoImage(im)
    imLabel = tk.Label(TK, image=img)
    imLabel.place(x=20, y=50)
    text1 = tk.Label(TK, text='                        ', font=('微软雅黑', 20))
    text1.place(x=500, y=300)
    text = tk.Label(TK, text=preds_result, font=('微软雅黑', 20))
    text.place(x=500, y=300)


def choose_fiel():
    selectFileName = tk.filedialog.askopenfilename(title='选择文件')  # 选择文件
    e.set(selectFileName)

e = tk.StringVar()
e_entry = tk.Entry(TK, width=45, textvariable=e)
e_entry.place(x=450, y=100)

submit_button = tk.Button(TK, text="选择文件", command=choose_fiel)
submit_button.place(x=710, y=130)

b = tk.Button(TK, text="开始识别", width=30, command=lambda: start(e_entry.get()))
b.place(x=500, y=180)

TK.mainloop()

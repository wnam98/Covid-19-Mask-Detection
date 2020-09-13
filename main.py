from tkinter import *
import numpy as np
from tkmacosx import Button
from ecapture import ecapture as ec
import time
import tensorflow as tf
from tensorflow import keras

new_model = tf.keras.models.load_model('mask_detector.model')
class_names = ['mask', 'no_mask']


def image_analysis():
    image_path = 'image.jpg'
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    return 100 * np.max(score) > 50


def clicked():
    while True:
        ec.capture(0, False, 'image.jpg')
        if image_analysis():
            break
        time.sleep(2)
    warning.config(fg='black', bg='red', text='Non-masked customer detected')


window = Tk()
window.title("COVID Guard")
window.geometry('500x500')

btn = Button(
    window,
    text="Click to Begin Surveillance",
    bg='white',
    fg='Black',
    borderless=1,
    font=("roboto", 20),
    command=clicked,
)
warning = Label(
    window,
    fg='spring green', bg='forest green', text='All Customers are Masked',
    height=40,
    width=100
)

btn.pack(fill='both')
warning.pack()
window.mainloop()

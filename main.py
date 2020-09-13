from tkinter import *
import cv2
import numpy as np
from tkmacosx import Button
from ecapture import ecapture as ec
import time


def image_analysis(path_to_image):

    img = cv2.imread(path_to_image)
    height, width, depth = img.shape
    circle_img = np.zeros((height, width), np.uint8)
    mask = cv2.circle(circle_img, (int(width / 2), int(height / 2)), 1, 1, thickness=-1)
    masked_img = cv2.bitwise_and(img, img, mask=circle_img)
    circle_locations = mask == 1
    bgr = img[circle_locations]
    rgb = bgr[..., ::-1]
    return rgb[0][0]


def clicked():
    while True:
        ec.capture(0, False, 'image.jpg')
        if image_analysis('image.jpg') > 72:
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

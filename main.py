from tkinter import *
from tkmacosx import Button
from ecapture import ecapture as ec
from boar.running import run_notebook
import time
outputs = run_notebook("image_detection.ipynb")


def clicked():
    while True:
        ec.capture(0, False, 'image.jpg')
        print('image taken')
        time.sleep(2)


window = Tk()
window.title("COVID Guard")
window.geometry('500x500')

font_color = 'black'
background_color = 'white'

btn = Button(
    window,
    text="Click to Begin Surveillance",
    bg='white',
    fg='Black',
    borderless=1,
    font=("roboto", 20),
    command=clicked
)
warning = Label(window, text='safe', fg=font_color, bg=background_color, height=40, width=20)

btn.pack()
warning.pack()
window.mainloop()

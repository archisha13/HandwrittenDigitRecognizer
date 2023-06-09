from tkinter import *
from tkinter import filedialog,messagebox,colorchooser
from PIL import Image,ImageDraw,ImageGrab
import PIL
from tkinter.font import Font
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


model=tf.keras.models.load_model('handwritten.model')
WIDTH,HEIGHT=500,500
CENTER=WIDTH//2
WHITE=(255,255,255)


class PaintGUI:

    def __init__(self):

        self.root=Tk()
        self.root.title('Digit Recognizer')

        self.my_font=Font(family="Just tell me what", size=25, weight="bold")

        self.brush_width=30
        self.current_color='#000000'

        self.cnv=Canvas(self.root,width=WIDTH-10,height=HEIGHT-10,bg='white')
        self.cnv.pack()

        self.cnv.bind("<B1-Motion>",self.paint)

        self.image=PIL.Image.new("RGB",(WIDTH,HEIGHT),WHITE)
        self.draw=ImageDraw.Draw(self.image)

        self.btn_frame=Frame(self.root)
        self.btn_frame.pack(fill=X)

        self.btn_frame.columnconfigure(0,weight=1)
        self.btn_frame.columnconfigure(1, weight=1)
        self.btn_frame.columnconfigure(2, weight=1)

        self.clear_btn=Button(self.btn_frame,text='Clear',command=self.clear)
        self.clear_btn.grid(row=0,column=1,sticky=W+E)

        self.save_btn = Button(self.btn_frame, text='Predict', command=self.save)
        self.save_btn.grid(row=1, column=1, sticky=W + E)

        # self.bplus_btn = Button(self.btn_frame, text='B+', command=self.brush_plus)
        # self.bplus_btn.grid(row=1, column=0, sticky=W + E)
        #
        # self.bminus_btn = Button(self.btn_frame, text='B-', command=self.brush_minus)
        # self.bminus_btn.grid(row=0, column=0, sticky=W + E)

        self.title_label=Label(self.root,text="AI digit recognizer",font=self.my_font,bg="white",fg="purple").place(x=100,y=20)
        #self.predict_label.grid(row=1,column=1,sticky=W+E)

        self.predict_label = Label(self.root, text="Prediction", font=("Elephant", 12), bg="#f0f0f0",fg="orange").place(x=380, y=492)

        self.text_val = StringVar()
        self.text_val.set('-')
        self.predict_text=Label(self.root,textvariable=self.text_val,font=("Arial",12)).place(x=410,y=520)

        # self.color_btn = Button(self.btn_frame, text='Change Color', command=self.change_color)
        # self.color_btn.grid(row=1, column=1, sticky=W + E)



        self.root.protocol('WM_DELETE_WINDOW',self.on_closing)
        self.root.attributes('-topmost',True)
        self.root.mainloop()


    def paint(self,event):
        x1,y1=(event.x-1),(event.y-1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.cnv.create_rectangle(x1,y1,x2,y2,outline=self.current_color,fill=self.current_color,width=self.brush_width)
        self.draw.rectangle([x1,y1,x2+self.brush_width,y2+self.brush_width],outline=self.current_color,fill=self.current_color,width=self.brush_width)

    def clear(self):
        self.cnv.delete('all')
        self.draw.rectangle([0,0,1000,1000],fill="white")



    def save(self):
        #filename=filedialog.asksaveasfilename(initialfile="untitled.png",defaultextension="png",filetypes=[("PNG","JPG"),(".png",".jpg")])

        filename2="images\\untitled.png"
        if filename2!="":
            self.image.save(filename2)

        im=Image.open("images\\untitled.png")
        im.thumbnail((28,28),Image.LANCZOS)
        im.save("images\\untitled.png")

        img = cv.imread("images\\untitled.png")[:,:,0]
        img = np.invert(np.array([img]))

        prediction = model.predict(img)
        print(prediction)
        self.text_val.set(np.argmax(prediction))
        self.root.update_idletasks()

        print("The digit is ", np.argmax(prediction))


        # if os.path.exists("D:\\pythonProject\\IBM_digit_recognizer\\images\\untitled.png"):
        #     os.remove("D:\\pythonProject\\IBM_digit_recognizer\\images\\untitled.png")
        # else:
        #     print("The file does not exist")
    def change_color(self):
        pass
    def on_closing(self):
        self.root.destroy()
        exit(0)
PaintGUI()
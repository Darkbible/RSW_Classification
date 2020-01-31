from tkinter import *
import tkinter as tk
from PIL import Image,ImageTk
import pandas as pd
import numpy as np
import glob
import os
import cv2
import tensorflow as tf


####################### CNN MODEL ####################################
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v

MobileNetV3_Small_Spec = [
    # Op            k    exp    out    SE     NL        s
    [ "ConvBnAct",  3,   False, 16,    False, "hswish", 2 ],
    [ "bneck",      3,   16,    16,    True,  "relu",   2 ],
    [ "bneck",      3,   72,    24,    False, "relu",   2 ],
    [ "bneck",      3,   88,    24,    False, "relu",   1 ],
    [ "bneck",      5,   96,    40,    True,  "hswish", 2 ],
    [ "bneck",      5,   240,   40,    True,  "hswish", 1 ],
    [ "bneck",      5,   240,   40,    True,  "hswish", 1 ],
    [ "bneck",      5,   120,   48,    True,  "hswish", 1 ],
    [ "bneck",      5,   144,   48,    True,  "hswish", 1 ],
    [ "bneck",      5,   288,   96,    True,  "hswish", 2 ],
    [ "bneck",      5,   576,   96,    True,  "hswish", 1 ],
    [ "bneck",      5,   576,   96,    True,  "hswish", 1 ],
    [ "ConvBnAct",  1,   False, 576,   True,  "hswish", 1 ],
    [ "pool",       7,   False, False, False, "None",   1 ],
    [ "ConvNBnAct", 1,   False, 1280,  False, "hswish", 1 ],
    [ "ConvNBnAct", 1,   False, 1000,  False, "None",   1 ],
]

from mobilenet_base import MobileNetBase
from MobileNetV3 import HardSigmoid
from MobileNetV3 import HardSwish
from MobileNetV3 import ConvBnAct
from MobileNetV3 import ConvNBnAct
from MobileNetV3 import BottleNeck
from MobileNetV3 import CusReshape
from MobileNetV3 import CusDropout
from MobileNetV3 import SENet
from MobileNetV3 import Pool

spec = MobileNetV3_Small_Spec

entry = tf.keras.layers.Input(shape=(224,224,1), name="inputs")

_available_operation = {
            "ConvBnAct":  ConvBnAct,
            "bneck":      BottleNeck,
            "pool":       Pool,
            "ConvNBnAct": ConvNBnAct,
        }

classes_number = 2
type =="small"

for i, params in enumerate(spec):
        Op, k, exp, out, SE, NL, s = params
        inference_op = _available_operation[Op]

        if isinstance(exp, int):
            exp_ch = _make_divisible(exp *1,8)
        else:
            exp_ch = None
        if isinstance(out, int):
            out_ch = _make_divisible(out * 1,8)
        else:
            out_ch = None
        if i == len(spec) - 1:  # fix output classes error.
            out_ch = classes_number

        op_name = f'{Op}_{i}'
        if i == 0:
            output = inference_op(k, exp_ch, out_ch, SE, NL, s,2e-5, op_name)(entry)
            outputQ = inference_op(k, exp_ch, 1, SE, NL, s,2e-5, op_name)(entry)
            
        else:
            output = inference_op(k, exp_ch, out_ch, SE, NL, s,2e-5, op_name)(output)
            outputQ = inference_op(k, exp_ch, 1, SE, NL, s,2e-5, op_name)(outputQ)
            

        if (type == "small" and i == 14) or (type == "large" and i == 18):
            output1 = CusDropout(dropout_rate=0.4)(output)
            output2 = CusDropout(dropout_rate=0.4)(outputQ)
            output = tf.keras.activations.linear(output1)
            #output = tf.keras.layers.ReLU(output1)
            #output =tf.keras.backend.relu(output1,alpha=0.0,max_value = maxval,threshold=0)
            outputQ = tf.keras.activations.sigmoid(output2)         
            outputQ = tf.nn.softmax(outputQ)
cnn_outputs = CusReshape(classes_number)(output)
Q_outputs   = outputQ
Q_outputs = CusReshape(2)(output)

####################### Setup #####################################################

PATH = 'C:/Users/darkb/Desktop/RSW_Kai _Mk3' #Place to save predict image

####################### GUI ##########################################

#Config
white 		= "#ffffff"
lightBlue2 	= "#adc5ed"
font 		= "Constantia"
fontButtons = (font, 12)
maxWidth  	= 680
maxHeight 	= 700

#Graphics window
mainWindow = tk.Tk()
mainWindow.configure(bg=lightBlue2)
mainWindow.geometry('%dx%d+%d+%d' % (maxWidth,maxHeight,0,0))
mainWindow.resizable(0,0)

# mainWindow.overrideredirect(1)

mainFrame = Frame(mainWindow)
mainFrame.place(x=20, y=20)                

#Capture video frames
lmain = tk.Label(mainFrame)
lmain.grid(row=0, column=0)

cap = cv2.VideoCapture(0)

####################### Main Loop ########################################################
def show_frame():
    ret, frame = cap.read()
    cv2image   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img   = Image.fromarray(cv2image).resize((640, 480))
    imgtk = ImageTk.PhotoImage(image = img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)
def Detect():
    ret, frame = cap.read()
    #frame = cv2.resize(frame,(64,64))
    cv2.imwrite(os.path.join(PATH,'Test'+'.png'),frame)
    
    test = cv2.imread('Test.png')
    test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    test = cv2.resize(test,(224,224))
    test = np.array(test)/255
    test = np.reshape(test,(1,224,224,1))
    
    model = tf.keras.Model(inputs = entry , outputs = cnn_outputs)
    model.load_weights('./model.h5')
    ans = model.predict(test)
    ans1 = ans[0,0]*280
    ans2 = ans[0,1]*2.8
    model2 = tf.keras.Model(inputs = entry , outputs = Q_outputs)
    model2.load_weights('./model2.h5')
    ans3 = model2.predict(test)
    
    if ans3 [0,0]>0.5:
        Q = 'Accept'
    if ans3 [0,1]>0.5:
        Q = 'Reject'
    #print(ans3)
    #print('Strenght(MPa):',ans1,'\t Diameter(mm):',ans2,'\t Quality:',Q)

    Text = 'Strenght(MPa): '+ str(round(ans1))+'  Diameter(mm): '+str(round(ans2))+'  Quality: '+str(Q)
    msg = tk.Text(mainWindow,height = 1, width = 50)
    msg.config(bg = lightBlue2 , font=('times',18, 'italic'))
    msg.insert(tk.END,Text)
    msg.config(state=DISABLED)
    msg.place(x=60,y= 580)   
    
def Close():
    cap.release()
    mainWindow.destroy()

######################################## Text & Button #######################################
Text = "Result :"
msg = tk.Message(mainWindow, text = Text)
msg.config(bg = lightBlue2 , font=('times',24, 'italic'))
msg.place(x=320,y=510)
    

closeButton = Button(mainWindow, text = "CLOSE", font = ('times',12, 'italic'),
                     bg = white, width = 20, height= 1)
closeButton.configure(command=Close)              
closeButton.place(x=100,y=650)

captureButton = Button(mainWindow, text = "DETECT", font = ('times',12, 'italic'),
                       bg = white, width = 20, height= 1)
captureButton.configure(command=Detect)              
captureButton.place(x=425,y=650)

show_frame()  #Display
mainWindow.mainloop()  #Starts GUI


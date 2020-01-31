import pandas as pd
import tensorflow as tf
import numpy as np
import glob
import os
import cv2
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Tn = pd.read_csv('Train.csv') #import traing data

def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v
#create data table for regression
data = Tn.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(Tn)),columns=['Amp','Time','Diameter(mm)','Strenght(MPa)'])

#create table with column Amp Time Diameter(mm) and Strenght(Mpa)
for i in range(0,len(data)):
    new_data['Amp'][i] = data['Amp'][i]
    new_data['Time'][i] = data['Time'][i]
    new_data['Diameter(mm)'][i] = data['Diameter(mm)'][i]
    new_data['Strenght(MPa)'][i] = data['Strenght(MPa)'][i]
#create data table for classification
Q_data = pd.DataFrame(index=range(0,len(Tn)),columns=['Amp','Time','Quality'])

#create table with column Amp Time and Quality
for i in range(0,len(data)):
    Q_data['Amp'][i] = data['Amp'][i]
    Q_data['Time'][i] = data['Time'][i]
    Q_data['Quality'][i] = data['Quality'][i]

# Make one-hot label
a = np.array(Q_data)
label0 = a[:,2] == 'Accept'
label1 = a[:,2] == 'Reject'
label = np.column_stack((label0,label1))
label = label.astype(int)

mylist = ["Time","Amp"]  #Categories of data
for i in mylist:
    print(i)
    encode_text_dummy(new_data,i)
    encode_text_dummy(Q_data,i)

new_images=[]
#add picture to "new_image" list
for number in range(0,len(new_data)+1):
    for path in glob.glob("./RSW_data/" + str(number) + ".jpg"):
        if os.path.isfile(path):
            new_images.append(path) 
img= pd.DataFrame(new_images[:1000],columns = ['image']) #Create dataframe with column name "image"

images_output=[]
for row_index,row in img[:len(img)].iterrows():    #resize every image found in "img" dataframe to 224*224
            inputImages=[]
            #outputImage = np.zeros((224, 224), dtype="uint8") 
            image_temp1 = cv2.imread(row.image)
            image_temp2 = cv2.resize(image_temp1, (224,224))
            image1 = cv2.cvtColor(image_temp2, cv2.COLOR_BGR2GRAY)
            
            inputImages.append(image1)
            outputImage = inputImages[0]
            images_output.append(outputImage)
images = np.array(images_output)/255

#create train set and test set for regression
x_text_train,x_text_test,x_images_train,x_images_test = train_test_split(new_data,images,test_size=0.3,random_state=42)

#create train set and test set for classification
x_text_train_Q,x_text_test_Q,x_images_train_Q,x_images_test_Q= train_test_split(label,test_size=0.3,random_state=42)


x_images_train_Q,x_images_train =np.reshape(x_images_train,(63,224,224,1))
x_images_test_Q,x_images_test = np.reshape(x_images_test,(27,224,224,1))

# Normalize data
maxval = x_text_train[['Strenght(MPa)','Diameter(mm)']].max() 
y_train = x_text_train[['Strenght(MPa)','Diameter(mm)']]/maxval
y_test = x_text_test[['Strenght(MPa)','Diameter(mm)']]/maxval

y_train = np.array(y_train)
y_test  = np.array(y_test)

y_train_Q = x_text_train_Q
y_test_Q = x_text_test_Q

################################MobileNetV3#####################

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

classes_number = 2 #number of regression class
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
            outputQ = inference_op(k, exp_ch, 2, SE, NL, s,2e-5, op_name)(entry)
            
        else:
            output = inference_op(k, exp_ch, out_ch, SE, NL, s,2e-5, op_name)(output)
            outputQ = inference_op(k, exp_ch, 2, SE, NL, s,2e-5, op_name)(outputQ)
            

        if (type == "small" and i == 14) or (type == "large" and i == 18):
            output1 = CusDropout(dropout_rate=0.4)(output)
            output2 = CusDropout(dropout_rate=0.4)(outputQ)
            output =tf.keras.backend.relu(output1,alpha=0.0,max_value = maxval,threshold=0) #regression
            
            outputQ = tf.keras.activations.sigmoid(output2) #classification
            outputQ = tf.nn.softmax(outputQ)
            
            
cnn_outputs = CusReshape(classes_number)(output)  #output for regression
Q_outputs   = outputQ
Q_outputs = CusReshape(2)(output) #output for classification

#loop for train regression by test sample and validation test
for i in range(1):
    print(i)
    model = tf.keras.Model(inputs = entry , outputs = cnn_outputs)
    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) 
    monitor1 = tf.keras.callbacks.TensorBoard(log_dir='logs',histogram_freq=0,update_freq='epoch',profile_batch=2)
    model.compile(loss="mean_squared_error",optimizer = adam)
    model.fit(
        [x_images_train], y_train, callbacks=[monitor1]
        ,validation_data=([x_images_test], y_test),epochs=300)
model.save_weights('model.h5')

#loop for train classification by test sample and validation test
for i in range(1):
    print(i)
    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model2 = tf.keras.Model(inputs = entry , outputs = Q_outputs)
    monitor2 = tf.keras.callbacks.TensorBoard(log_dir='logs_Q',histogram_freq=0,update_freq='epoch',profile_batch=2)
    model2.compile(loss="mean_squared_error",optimizer = adam)
    model2.fit([x_images_train_Q], y_train_Q, callbacks=[monitor2]
        ,validation_data=([x_images_test_Q], y_test_Q),epochs=300)
model2.save_weights('model2.h5')

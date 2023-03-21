
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense , Conv2D , MaxPool2D , Flatten , Dropout
from keras.utils import np_utils
from sklearn.metrics import accuracy_score

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()   #Split the dataset into test and train data.

# x_train=tf.keras.utils.normalize(x_train,axis=1)
# x_test=tf.keras.utils.normalize(x_test,axis=1)   #Normalizes a Numpy array and axis gives the axis along which we normalize.

sh = x_train.shape
train_samples = sh[0]
x_train = x_train.reshape(train_samples,28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)


numb_classes =10
# ohe - one hot encoding
y_train_ohe = np_utils.to_categorical(y_train , numb_classes)
y_test_ohe = np_utils.to_categorical(y_test , numb_classes)
print(y_train_ohe)

model = Sequential()
#Image filtered by 25 filters
model.add(Conv2D(25,kernel_size=(3,3),strides=(1,1) , padding = 'valid', activation ='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size =(2,2)))
model.add(Flatten())
#fully connected layer having 100 neurons
model.add(Dense(100,activation ='relu'))
model.add(Dense(10,activation ='softmax'))

model.compile(loss ='categorical_crossentropy', metrics =['accuracy'],optimizer = 'adam')
#model will train to assure there is higher accuracy

model.fit(x_train,y_train_ohe, batch_size =128 , epochs =5)


model.save('handwritten.model')         #saving the model


#
# #--------------------------------------------------------------------------------------------------#
# model=tf.keras.models.load_model('handwritten.model')   #loading the saved model
# #loss,accuracy=model.evaluate(x_test,y_test)             #calculating accuracy and loss of data
# # print(loss)
# # print(accuracy)
#
#
# img=cv.imread("images\\untitled.png")[:,:,0]   #loading the image saved for testing
# print(img.shape)                               #checking the shape(pixel ratio) of the testing image
# # plt.imshow(img,cmap=plt.cm.binary)
# img=np.invert(np.array([img]))                 #inverting image colors
#
# prediction=model.predict(img)                   #prediction
# #print(prediction)
# print("The digit is ",np.argmax(prediction))    #np.argmax returns the maximum value found from the prediction array
# plt.imshow(img[0],cmap=plt.cm.binary)           #displaying the test image
# # plt.imshow(x_train[1],cmap=plt.cm.binary)     #sample data from dataset
# plt.show()

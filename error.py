#TODO: writing a program to get predictive results from a linear input dataset

from keras.layers import Sequential
from keras.layers import Dense,Activation
import numpy

model=Sequential()
model.add(Dense(32,activation='relu',input_dim=20))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])


#dataset with independent vector ranging 1 to 20 and dependent vector 2 to 41 with help of numpy

x_train=numpy.arange(0,21,1)
y_train=numpy.arange(0,41,2)

x_test=numpy.arange(0,50,5)

#fitting the model
model.fit(x_train,y_train,epochs=10,batch_size=32)



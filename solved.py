#TODO: writing a program to get predictive results from a linear input dataset

from keras.models import Sequential
from keras.layers import Dense,Activation
import numpy

model=Sequential()
#model.add(Dense(20,activation='relu',input_dim=20))
#model.add(Dense(1,activation='sigmoid'))
model.add(Dense(20,input_dim=20))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])


#dataset with independent vector ranging 1 to 20 and dependent vector 2 to 41 with help of numpy

x_train=numpy.arange(1,21,1)
y_train=numpy.arange(1,41,2)

x_test=numpy.arange(0,50,5)

x_train=x_train.reshape([1,20])
y_train=y_train.reshape([1,20])

#fitting the model

#model.fit(x_train,y_train,epochs=10,batch_size=32)
model.fit(x_train,y_train)
y=model.predict(x_train)

print y



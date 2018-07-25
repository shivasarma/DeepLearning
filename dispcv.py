# TODO: We are implementing deep learning mnist database. The MNIST database includes digits and alphabets. The final output should be the indicatoin of what the given character. Anyhow as on july 24 i am trying to display the character.


## libraries... importing mnist. MNIST is minimal package that doesn't contain any details when tried with help.

##importing os library to migrate the directory from one place to another place

##the code is copied and referred from https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python
#https://stackoverflow.com/questions/46689428/convert-np-array-of-type-float64-to-type-uint8-scaling-values
#https://askubuntu.com/questions/342202/failed-to-load-module-canberra-gtk-module-but-already-installed


##importing opencv libraries to view it graphically. Required to display it graphically

from mnist import MNIST
import os
import cv2
import numpy

#os.chdir('.') this is intended to change the directory from a different directory to the current directory which is obselete since the same directory doesn't require because already the path is given. However with the following commented code we can see all the files stored

'''

for file in os.listdir(os.curdir):
    print file'''

mndata = MNIST(os.getcwd()) # here '.' and os.getcwd gives the same output which is the current working directory.
images,labels=mndata.load_training()
#images,labels=mndata.load_testing()

#the images here is are of list type. That needs to be converted to numpy array with uint8 variables.

image=images[5]
image = numpy.resize(image,(28,28))   # Please check the documentation of MNIST dataset to get details of the resize feature
imagecv=image.astype(numpy.uint8)
cv2.imshow('img',imagecv)
cv2.waitKey()
print(mndata.display(images[5])) # this displays the 5th image in the array.


##Conclusion we can do it, still we are not able to load the dataset and experiment.



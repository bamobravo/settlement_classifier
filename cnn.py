import os; os.environ['KERAS_BACKEND'] = 'theano'
from skimage.transform import resize
from skimage.io import imread,imshow,imsave
import skimage.io as imageio
import numpy as np
import os
import sys
import glob
from scipy import misc
import matplotlib.pyplot as plt
from keras.models import model_from_json

#import for cnn modeling
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

def isImage(filename):
    filename = filename.lower()
    fl =['jpeg','jpg','png','gif']
    ps = filename.rfind('.')
    if ps==-1:
        return False
    ext = filename[ps+1:]
    return ext in fl
    
def saveModel(model):
    jsonModel = model.to_json()
    with open("model.json",'w') as json_file:
        json_file.write(jsonModel)
    # save the weight
    model.save_weights('model.h5')
    print("model saved successfully")


def loadModel():
    fl = open('model.json','r')
    loaded_json = fl.read()
    fl.close()
    model = model_from_json(loaded_json)
    model.load_weights('model.h5')
    return model

def predict(model,filename):
    #convert the image into the right size,
    size=(200,200)
    img = misc.imread(filename, mode='RGB')
    img = resize(img,size)
    result = np.asarray([img])
    result = result/255
    return model.predict(result)


def rescaleImage(sourcePath,destinationPath,size):
    '''
    rescaled the set of images in this folder and add them all to a different folder
    '''
    if not os.path.isdir(sourcePath):
        exit('cannot proceed , source path is not a directory')
    
    # create the source directory
    if not os.path.isdir(destinationPath):
        # create the directory
        os.mkdir(destinationPath)
    
    files = os.listdir(sourcePath)
    for fl  in files:
        if isImage(fl):
            print("Resizing file ",fl)
            img = imread(os.path.join(sourcePath,fl))
            img = resize(img,size)
            ind = fl.rfind('.')
            newFile = fl[:ind]+".png"
            newPath = os.path.join(destinationPath,newFile)
            imsave(newPath,img)



def cnn(size, n_layers):
    # INPUTS
    # size     - size of the input images
    # n_layers - number of layers
    # OUTPUTS
    # model    - compiled CNN

    # Define hyperparamters
    MIN_NEURONS = 5
    MAX_NEURONS = 60
    KERNEL = (3, 3)

    # Determine the # of neurons in each convolutional layer
    steps = np.floor(MAX_NEURONS / (n_layers + 1))
    nuerons = np.arange(MIN_NEURONS, MAX_NEURONS, steps)
    nuerons = nuerons.astype(np.int32)

    # Define a model
    model = Sequential()

    # Add convolutional layers
    for i in range(0, n_layers):
        if i == 0:
            shape = (size[0], size[1], size[2])
            model.add(Conv2D(nuerons[i], KERNEL, input_shape=shape))
        else:
            model.add(Conv2D(nuerons[i], KERNEL))

        model.add(Activation('relu'))

    # Add max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(MAX_NEURONS))
    model.add(Activation('relu'))

    # Add output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Print a summary of the model
    model.summary()
    return model

#function to load the data needed
def loadData(folderPath,label):
    file_paths = glob.glob(os.path.join(folderPath, '*.png'))
    images = [misc.imread(path,mode='RGB') for path in file_paths]
    images = np.asarray(images)
    return images/255,np.array([label ]*images.shape[0])

def splitData(images,labels):

    # Split into test and training sets
    TRAIN_TEST_SPLIT = 0.9

    # Split at the given index
    split_index = int(TRAIN_TEST_SPLIT * images.shape[0])
    shuffled_indices = np.random.permutation(images.shape[0])
    train_indices = shuffled_indices[0:split_index]
    test_indices = shuffled_indices[split_index:]
    # Split the images and the labels
    x_train = images[train_indices, :, :]
    y_train = labels[train_indices]
    x_test = images[test_indices, :, :]
    y_test = labels[test_indices]
    return x_train,y_train,x_test,y_test


def doTraining():
    data_positive, label_positive = loadData('positive', 1)
    data_negative, label_negative = loadData('negative', 0)
    all_data = np.concatenate((data_positive, data_negative),
                              axis=0) if len(data_negative) > 0 else data_positive
    all_label = np.concatenate(label_positive, label_negative) if len(
        data_negative) > 0 else label_positive
    X, Y, x, y = splitData(all_data, all_label)

    print(X.shape[0])
    #generate the test and trainning data

    layer = 2
    image_dim = np.asarray([X.shape[1], X.shape[2], X.shape[3]])
    model = cnn(image_dim, layer)

    epochs = 50
    batch = 8
    pat = 10
    early_stopping = EarlyStopping(
        monitor='loss', min_delta=0, patience=pat, verbose=0, mode='auto')
    model.fit(X, Y, epochs, batch, verbose=0)
    #evaluate the model from here
    prediction=model.predict(x)
    print(prediction)
    acc = accuracy_score(y,prediction)
    print("Accuracy: "+str(acc))

    saveModel(model)

#check that model already exists
if os.path.isfile('model.h5') and os.path.isfile('model.json'):
    val =input("trained model found will you like to use the saved model?(y/n)")
    val = val.lower()
    if val not in ['y','yes']:
        print("retraining model")
        print()
        doTraining()
else:
    doTraining()

model = loadModel()
print()
stillRunning=True
threshold=0.7
diseaseName='schizo'
print("Welcome, this program test an image is of a "+diseaseName+" \n You use specifying the path to the image to be tested and the program generate the output.")
while stillRunning:
    print()
    path = input("enter the path of image to the test for "+diseaseName+" (type q to exit):")
    if path.lower()=='q':
        break
    result = predict(model, path)
    if result[0]>threshold:
        print("Yes, "+diseaseName+" is present!")
        print()
    else:
        print("No, "+diseaseName+" is not present!")
        print()
print("Thanks for using the program, bye.")



# if __name__=="__main__":
#     if len(sys.argv) < 4:
#         print("usage: You have to supply the following parameters\n supply the source folder(optional is running from the current folder),\n The destination folder \n scale width and \n scale height")
#         exit()
#     rescaleImage(sys.argv[1], sys.argv[2], (int(sys.argv[3]), int(sys.argv[4])))

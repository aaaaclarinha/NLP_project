# %% [markdown]
# # Tutorials 98, 99 e 100

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

plt.style.use('classic')

import tensorflow.keras as keras
import tensorflow as tf
from keras import backend 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.datasets import cifar10
from keras.utils import normalize, to_categorical
from keras.layers import Dropout
from keras.optimizers import SGD, RMSprop

# %%
### Normalize inputs
#WHat happens if we don't normalize inputs?
# ALso we may have to normalize depending on the activation function

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print("The size of training dataset is: ", X_train.shape)
print("The size of testing dataset is: ", X_test.shape)

# %%
#Decrease the dataset size to see the effect - decrease it to 1000 (test_size=0.1)
from sklearn.model_selection import train_test_split
_, X, _, Y = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)
print("The size of the dataset X is: ", X.shape)
print("The size of the dataset Y is: ", Y.shape)

# %%
#Split again into train and test to create small training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
print("The size of training dataset is: ", X_train.shape)
print("The size of testing dataset is: ", X_test.shape)
print("The size of training dataset y is: ", y_train.shape)
print("The size of testing dataset y is: ", y_test.shape)

# %%
#view few images 
for i in range(9):
	plt.subplot(330 + 1 + i)
	plt.imshow(X_train[i])
plt.show()

# %%
X_train = (X_train.astype('float32')) / 255.
X_test = (X_test.astype('float32')) / 255.

# %%
# Print a few y_train values to see before and after categorical
print(y_train[0])
print(y_train[1])
print(y_train[10])

# %%
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# %%
print(y_train[0])
print(y_train[1])
print(y_train[10])

# %%
# Create a model with dropout
drop=0.25

#Kernel = zeros --> No change in weights... like vanishing gradient problem
#kernel = random --> Performs better but when you rerun the experiment the results may vary quite a bit, depends on the application. 
#kernel = he_uniform --> Ideal to work with relu. 
#kernel = glorot_uniform --> similar to he_uniform but different variance. he_uniform is preferred with ReLu

kernel_initializer =  'he_uniform'  #Also try 'zeros', 'random_uniform', 'he_uniform', 'glorot_uniform'

model1 = Sequential()
model1.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', input_shape=(32, 32, 3)))
model1.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Dropout(drop))

model1.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))
model1.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Dropout(drop))

model1.add(Flatten())
model1.add(Dense(512, activation='relu', kernel_initializer=kernel_initializer))
model1.add(Dropout(drop))
model1.add(Dense(10, activation='softmax'))

opt1 = SGD(learning_rate=0.001, momentum=0.9)
opt2 = RMSprop(learning_rate=0.001)
model1.compile(optimizer=opt1, loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()

# %%
#########################################################
#Fit model....
history = model1.fit(X_train, y_train, epochs=25, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# %%
#####################################################################
#plot the training and validation accuracy and loss at each epoch
#If validation loss is lower than training loss this could be becuase we are applying
#regularization (Dropout) during training which won't be applied during validation. 
#Also, training loss is measured during each epoch while validation is done after the epoch. 

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# %% [markdown]
# # Tutorial 101

# %% [markdown]
# ### Não achei muito interessante. Porém o vídeo explica muito bem os gráficos obtidos acima.

# %% [markdown]
# # Tutorial 102 - Validation Dataset

# %%
#Let us extract only the 50000 training data available from cifar for this exercise.  
(X0, Y0), (_, _) = cifar10.load_data()
print("The size of dataset X is: ", X.shape)  #Images
print("The size of dataset Y is: ", Y.shape)  #Corresponding labels

# %%
_, X, _, Y = train_test_split(X0, Y0, test_size = 0.2, random_state = 0)
print("The size of the dataset X is: ", X.shape)
print("The size of the dataset Y is: ", Y.shape)

# %%
#Holding out 10% of all data to be used for testing purposes. 
#This data will never see the training. 
from sklearn.model_selection import train_test_split

X1, X_test, Y1, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)
print("The size of the test dataset is: ", X_test.shape)
print("The size of the remaining dataset is : ", X1.shape)

# %%
#Split again into train and test to create small training and testing dataset
X_train, X_valid, y_train, y_valid = train_test_split(X1, Y1, test_size = 0.25, random_state = 0)
print("The size of training dataset is: ", X_train.shape)
print("The size of testing dataset is: ", X_valid.shape)
print("The size of training dataset y is: ", y_train.shape)
print("The size of testing dataset y is: ", y_valid.shape)

# %%
#view few images 
for i in range(9):
	plt.subplot(330 + 1 + i)
	plt.imshow(X_train[i])
plt.show()

# %%
X_train = (X_train.astype('float32')) / 255.
X_valid = (X_valid.astype('float32')) / 255.
X_test = (X_test.astype('float32')) / 255.

# %%
# Print a few y_train values to see before and after categorical
print(y_train[0])
print(y_train[1])
print(y_train[10])

# %%
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
y_test = to_categorical(y_test)

# %%
print(y_train[0])
print(y_train[1])
print(y_train[10])

# %%
# Create a model with dropout
drop=0.25
kernel_initializer =  'he_uniform'  #Also try 'zeros', 'random_uniform', 'he_uniform', 'glorot_uniform'

model1 = Sequential()
model1.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', input_shape=(32, 32, 3)))
model1.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Dropout(drop))

model1.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))
model1.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Dropout(drop))

model1.add(Flatten())
model1.add(Dense(512, activation='relu', kernel_initializer=kernel_initializer))
model1.add(Dropout(drop))
model1.add(Dense(10, activation='softmax'))

opt1 = SGD(learning_rate=0.001, momentum=0.9)
opt2 = RMSprop(learning_rate=0.001, decay=1e-6)
model1.compile(optimizer=opt1, loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()

# %%
#########################################################
#Fit model....
history = model1.fit(X_train, y_train, epochs=25, batch_size=64, validation_data=(X_valid, y_valid), verbose=1)

# %%
#Evaluate the model against test data that never saw the training process. 
_, test_acc = model1.evaluate(X_test, y_test)
_, valid_acc = model1.evaluate(X_valid, y_valid)
print("Accuracy on the validation dataset = ", (valid_acc * 100.0), "%")
print("Accuracy on the test dataset = ", (test_acc * 100.0), "%")

# %%
#####################################################################
#plot the training and validation accuracy and loss at each epoch
#If validation loss is lower than training loss this could be becuase we are applying
#regularization (Dropout) during training which won't be applied during validation. 
#Also, training loss is measured during each epoch while validation is done after the epoch. 


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# %%
# Predicting the Test set results
y_pred_test = model1.predict(X_test)
prediction_test = np.argmax(y_pred_test, axis=1)
ground_truth = np.argmax(y_test, axis=1)

# %%
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(ground_truth, prediction_test)

sns.heatmap(cm, annot=True)

# %% [markdown]
# # Tutorial 103 - Data augmentation

# %%
from keras.preprocessing.image import array_to_img, img_to_array, load_img 
from skimage import io
from numpy import expand_dims
from matplotlib import pyplot as plt

# %%
# Construct an instance of the ImageDataGenerator class
# Pass the augmentation parameters through the constructor. 
datagen = ImageDataGenerator( 
        rotation_range = 45,      #Random rotation between 0 and 45
        width_shift_range=[-20,20],  #min and max shift in pixels
        height_shift_range=0.2,  #Can also define as % shift (min/max or %)
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True, 
        brightness_range = (0.5, 1.5), fill_mode='constant') #Values less than 1 darkens and greater brightens

# %%
# Loading a sample image  
#Can use any library to read images but they need to be in an array form
x = io.imread('images/monalisa.jpg')  #Array with shape (256, 256, 3)

# %%
# Reshape the input image 
x = x.reshape((1, ) + x.shape)  #Array with shape (1, 256, 256, 3)

# %%
#Let us save images to get a feel for the augmented images.
#Create an iterator either by using image dataset in memory (using flow() function)
#or by using image dataset from a directory (using flow_from_directory)
#from directory can beuseful if subdirectories are organized by class
   
# Generating and saving 10 augmented samples  
# using the above defined parameters.  
#Again, flow generates batches of randomly augmented images
i = 0
for batch in datagen.flow(x, batch_size = 1, 
                          save_to_dir ='images/augmented_monalisa',  
                          save_prefix ='aug', save_format ='jpeg'): 
    
    
    i += 1
    if i > 20: 
        break

# %% [markdown]
# ### Confira o diretório images/augumented_monalisa

# %% [markdown]
# ### Vamos voltar ao dataset cifar10

# %% [markdown]
# Já temos nosso train/test/validation split feito.

# %%
#view few images 
for i in range(9):
	plt.subplot(330 + 1 + i)
	plt.imshow(X_train[i])
plt.show()

# %%
# Create a model with dropout
drop=0.25

kernel_initializer =  'he_uniform'  #Also try 'zeros', 'random_uniform', 'he_uniform', 'glorot_uniform'

model1 = Sequential()
model1.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', input_shape=(32, 32, 3)))
model1.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Dropout(drop))

model1.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))
model1.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Dropout(drop))

model1.add(Flatten())
model1.add(Dense(512, activation='relu', kernel_initializer=kernel_initializer))
model1.add(Dropout(drop))
model1.add(Dense(10, activation='softmax'))

opt1 = SGD(learning_rate=0.001, momentum=0.9)
opt2 = RMSprop(learning_rate=0.001)
model1.compile(optimizer=opt1, loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()

# %%
###################################################
######### Data augmentation to improve the model

train_datagen = ImageDataGenerator(rotation_range=15,  #Too much rotation may hurt accuracy, especially for small datasets.
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range = 0.1,
    vertical_flip=False,
    horizontal_flip = True,
    fill_mode="reflect")

#train_datagen.fit(X_train)

train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size = 32)  #images to generate in a batch

# %%
x = next(train_generator)
print(x[0].shape)  #Images
print(x[1].shape)  #Labels
print((x[0].shape[0]))

# %%
x = next(train_generator)
image = x[0][0]
title = np.argmax(x[1][0])
plt.figure(figsize=(1.5, 1.5))
plt.suptitle(title, fontsize=12)
plt.imshow(image)
plt.show()

# %%
print("Total number of training images in the dataset = ", X_train.shape[0])

# %%
#NOTE: When we use fit_generator, the number of samples processed 
#for each epoch is batch_size * steps_per_epochs. 
#should typically be equal to the number of unique samples in our 
#dataset divided by the batch size.

batch_size = 32   #Match this to the batch_size from generator
steps_per_epoch = len(X_train) // batch_size  

print("Steps per epoch = ", steps_per_epoch)
print("Total data per epoch = ", steps_per_epoch*batch_size)

# %%
history = model1.fit(
        train_generator,
        steps_per_epoch = steps_per_epoch,
        epochs = 25,
        batch_size=64,
        validation_data = (X_valid, y_valid))

# %%
_, test_acc = model1.evaluate(X_test, y_test)
_, valid_acc = model1.evaluate(X_valid, y_valid)
print("Accuracy on the validation dataset = ", (valid_acc * 100.0), "%")
print("Accuracy on the test dataset = ", (test_acc * 100.0), "%")

# %%
#Accuracy with and without Augmentation
import pandas as pd
without_aug = {1000:36.4, 2000:45.2, 5000:51.7, 10000:58.4, 25000:69.4, 50000:77.3}
with_aug = {1000:44, 2000:48.4, 5000:54.7, 10000:60.8, 25000:70.7, 50000:78.4}
df = pd.DataFrame([without_aug, with_aug])
df = df.T
df.reset_index(inplace=True)


df.columns =['num_images', 'without_aug', 'with_aug']
print(df.head)

df.plot(x='num_images', y=['without_aug', 'with_aug'], kind='line')

# %%
#####################################################################
#plot the training and validation accuracy and loss at each epoch
#If validation loss is lower than training loss this could be becuase we are applying
#regularization (Dropout) during training which won't be applied during validation. 
#Also, training loss is measured during each epoch while validation is done after the epoch. 

history = history

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# %%
# Predicting the Test set results
y_pred_test = model1.predict(X_test)
prediction_test = np.argmax(y_pred_test, axis=1)
ground_truth = np.argmax(y_test, axis=1)

# %%
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(ground_truth, prediction_test)

sns.heatmap(cm, annot=True)

# %% [markdown]
# # Tutorial 104

# %%
# Create a model with dropout
drop=0.5  #Setting to 0.5 so the training stops when it encounters overfitting. 

#Kernel = zeros --> No change in weights... like vanishing gradient problem
#kernel = random --> Performs better but when you rerun the experiment the results may vary quite a bit, depends on the application. 
#kernel = he_uniform --> Ideal to work with relu. 
#kernel = glorot_uniform --> similar to he_uniform but different variance. he_uniform is preferred with ReLu

kernel_initializer =  'he_uniform'  #Also try 'zeros', 'random_uniform', 'he_uniform', 'glorot_uniform'

model1 = Sequential()
model1.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', input_shape=(32, 32, 3)))
model1.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Dropout(drop))

model1.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))
model1.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Dropout(drop))

model1.add(Flatten())
model1.add(Dense(512, activation='relu', kernel_initializer=kernel_initializer))
model1.add(Dropout(drop))
model1.add(Dense(10, activation='softmax'))

opt1 = SGD(learning_rate=0.001, momentum=0.9)
opt2 = RMSprop(learning_rate=0.001)
model1.compile(optimizer=opt1, loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()

# %%
#Add Callbacks, e.g. ModelCheckpoints, earlystopping, csvlogger.
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

#ModelCheckpoint callback saves a model at some interval. 

#Give unique name to save all models as accuracy improves
filepath="saved_models/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.keras" #File name includes epoch and validation accuracy.

#Overwrite the model each time accuracy improves. Saves a lot of space. 
#filepath="/content/drive/MyDrive/Colab Notebooks/saved_models/best_model.hdf5" #File name includes epoch and validation accuracy.
#Use Mode = max for accuracy and min for loss. 
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

#https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
#This callback will stop the training when there is no improvement in
# the validation loss for three consecutive epochs.

#CSVLogger logs epoch, accuracy, loss, val_accuracy, val_loss. So we can plot later.
log_csv = CSVLogger('saved_logs/my_logs.csv', separator=',', append=False)

callbacks_list = [checkpoint, early_stop, log_csv]

# %%
#########################################################
#Fit model....

history1 = model1.fit(X_train, y_train, epochs=50, batch_size=64, validation_data = (X_valid, y_valid), verbose=1, callbacks=callbacks_list)

# %%
_, test_acc = model1.evaluate(X_test, y_test)
_, valid_acc = model1.evaluate(X_valid, y_valid)
print("Accuracy on the validation dataset = ", (valid_acc * 100.0), "%")
print("Accuracy on the test dataset = ", (test_acc * 100.0), "%")

# %%
#####################################################################
#plot the training and validation accuracy and loss at each epoch
#If validation loss is lower than training loss this could be becuase we are applying
#regularization (Dropout) during training which won't be applied during validation. 
#Also, training loss is measured during each epoch while validation is done after the epoch. 

history = history1

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# %% [markdown]
# # Tutorial 105

# %%
# Create a model with dropout
drop=0.25

#Kernel = zeros --> No change in weights... like vanishing gradient problem
#kernel = random --> Performs better but when you rerun the experiment the results may vary quite a bit, depends on the application. 
#kernel = he_uniform --> Ideal to work with relu. 
#kernel = glorot_uniform --> similar to he_uniform but different variance. he_uniform is preferred with ReLu

kernel_initializer =  'he_uniform'  #Also try 'zeros', 'random_uniform', 'he_uniform', 'glorot_uniform'

model1 = Sequential()
model1.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', input_shape=(32, 32, 3)))
model1.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Dropout(drop))

model1.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))
model1.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Dropout(drop))

model1.add(Flatten())
model1.add(Dense(512, activation='relu', kernel_initializer=kernel_initializer))
model1.add(Dropout(drop))
model1.add(Dense(10, activation='softmax'))

opt = SGD(learning_rate=0.001, momentum=0.9)
model1.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#model1.summary()

# %%
#Functions for learning rate change

# This function keeps the initial learning rate for the first ten epochs  
# and decreases it exponentially after that.  
import math
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * math.exp(-0.1)

# Step decay
# Start with initial learning rateand drop LR by a factor after certain epochs.
decay_rate=[]
def step_decay(epoch):
  init_lrate = 0.1  #Initial LR
  drop = 0.05  #Drop factor
  epochs_drop = 2.0  # Number of epochs after which LR drops
  lr = init_lrate * math.pow(drop, math.floor((1+epoch)/(epochs_drop)))
  decay_rate.append(lr)
  return lr

decay_rate=[]
def exp_decay(epoch):
   initial_lrate = 0.1
   k = 0.1
   lr = initial_lrate * math.exp(-k*epoch)
   decay_rate.append(lr)
   return lr

#NOTE: Adam optimizer uses adaptive learning rate. So you may not need to schedule learning rate unless you use SGD. 

# %%
#Add Callbacks, e.g. ModelCheckpoints, earlystopping, csvlogger.
from keras.callbacks import LearningRateScheduler

#callbacks_list = [LearningRateScheduler(scheduler)]
callbacks_list = [LearningRateScheduler(exp_decay)]

#########################################################
#Fit model....

history1 = model1.fit(X_train, y_train, epochs=25, batch_size=64, validation_data = (X_valid, y_valid), verbose=1, callbacks=callbacks_list)


# %%
print("Decay rate :", decay_rate)
plt.plot(decay_rate)

# %%
_, test_acc = model1.evaluate(X_test, y_test)
_, valid_acc = model1.evaluate(X_valid, y_valid)
print("Accuracy on the validation dataset = ", (valid_acc * 100.0), "%")
print("Accuracy on the test dataset = ", (test_acc * 100.0), "%")

# %%
#####################################################################
#plot the training and validation accuracy and loss at each epoch
#If validation loss is lower than training loss this could be becuase we are applying
#regularization (Dropout) during training which won't be applied during validation. 
#Also, training loss is measured during each epoch while validation is done after the epoch. 

history = history1

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# %% [markdown]
# # Tutorial 106 - Assistir o vídeo, explica bem curva ROC e AUC.

# %% [markdown]
# https://www.youtube.com/watch?v=jbeATQXKtzw&list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG&index=110



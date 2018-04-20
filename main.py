from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

BATCH_SIZE = 64
IMAGE_SIZE = 150
MODE = 'binary'
TRAIN_PATH = 'data/train2'
VAL_PATH = 'data/validation'
TEST_PATH = 'data/test'
ERROR_PATH = 'data/error'
TRAIN_SAMPLES = 17500 +85
VAL_SAMPLES = 7500
TEST_SAMPLES = 1380 + 861
ERROR_SAMPLES = 58 + 27
SAVE_FILE = 'weights1.h5'
LOAD_FILE = 'weights1.h5'
EPOCH = 10

def train(train, val, epochs_train, save_file):
    model.fit_generator(
        train_generator,
        steps_per_epoch = TRAIN_SAMPLES // BATCH_SIZE,
        epochs=epochs_train,
        validation_data = val_generator,
        validation_steps = VAL_SAMPLES // BATCH_SIZE)
    model.save_weights(save_file)

def showErrors(generator, path, samples, batch, start, end):
    count = 0
    for i in zip(generator.filenames[start:end], model.predict_generator(generator).reshape(-1)[start:end]):
        text = '{0:45}   {1:1.4f}'.format(i[0], i[1])
        if ((i[0][0:4] == 'dogs' and i[1] < 0.5) or (i[0][0:4] == 'cats' and i[1] > 0.5)):
            img = Image.open(path + '\\' + i[0])
            plt.figure()
            plt.axis('off')
            plt.text(0,-20, text, size = 15)
            plt.imshow(img)
            count+=1
    print(count)
    
def showStat(generator, samples, batch):
    scores = model.evaluate_generator(generator, samples // batch)
    print(scores)

def createGenerator(path, x, batch, class_mode, shuffle):
    return datagen.flow_from_directory(
            path,
            target_size=(x, x),
            batch_size=batch,
            shuffle=shuffle,
            class_mode=class_mode)

datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = createGenerator(TRAIN_PATH, IMAGE_SIZE, BATCH_SIZE, MODE, True)
val_generator = createGenerator(VAL_PATH, IMAGE_SIZE, BATCH_SIZE, MODE, True)
test_generator = createGenerator(TEST_PATH, IMAGE_SIZE, BATCH_SIZE, MODE, False)
error_generator = createGenerator(ERROR_PATH, IMAGE_SIZE, BATCH_SIZE, MODE, False)

def model1():
    
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    return model

model = model1()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.load_weights(LOAD_FILE)

train(train_generator, val_generator, EPOCH, SAVE_FILE)
#showErrors(test_generator, TEST_PATH, VAL_SAMPLES, BATCH_SIZE, 1800, 2241)
#showStat(error_generator, ERROR_SAMPLES, BATCH_SIZE)

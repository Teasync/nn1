from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras import models, optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import backend as K
from PIL import Image
import os
import numpy as np
from datetime import datetime

user_profile = os.environ['USERPROFILE']

dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.mkdir('checkpoints/' + dt)
curdir = 'checkpoints/' + dt + '/'

def load_img(path):
    temp_img_ = Image.open(path)
    temp_img_.load()
    temp_img_ = temp_img_.resize((150, 150))
    temp_img_ = np.asarray(temp_img_)
    temp_img_ret = temp_img_ / 255.
    temp_img_ret = np.expand_dims(temp_img_ret, 0)
    return temp_img_ret


def load_model(name):
    with open(name, 'r') as json:
        json_contents = json.read()
    m = models.model_from_json(json_contents)
    return m


def load_weights(model_, name):
    model_.load_weights(name)


def save_model(model_, name):
    json = model_.to_json()
    with open(name, 'w') as openFile:
        openFile.write(json)


def save_weights(model_, name):
    model_.save_weights(name)


def new_model():
    model = Sequential()
    model.add(Conv2D(30, 3, input_shape=(150, 150, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(30, 3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Conv2D(60, 3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(60, 3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(120, 3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(120, 3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(240, 3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(240, 3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(480, 3, padding='same'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Activation('relu'))
    # model.add(Conv2D(480, 3, padding='same'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Activation('relu'))
    # model.add(Conv2D(480, 3, padding='same'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Nadam(lr=0.005),
                  metrics=['accuracy'])
    return model


def train_model(model, batch_size=100, steps_per_epoch=1000, epochs=1, callbacks=None):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(r'.\img\all\train',
                                                        target_size=(150, 150),
                                                        batch_size=batch_size,
                                                        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(r'.\img\all\validation',
                                                            target_size=(150, 150),
                                                            batch_size=batch_size,
                                                            class_mode='binary')

    return model.fit_generator(train_generator,
                               steps_per_epoch=steps_per_epoch,
                               epochs=epochs,
                               validation_data=validation_generator,
                               validation_steps=1000,
                               callbacks=callbacks)


class Classifier:
    def __init__(self, trained_model):
        self.__model__ = trained_model

    def get_model(self):
        return self.__model__

    def predict(self, *kwords):
        return


class CatDogClassifier(Classifier):
    def __init(self, trained_model):
        super().__init__(trained_model)

    def predict(self, filepath):
        val_ = self.__model__.predict(load_img(filepath))[0][0]
        if 0 <= val_ <= 0.3:
            print('Probably a cat', end='')
        elif 0.3 < val_ < 0.7:
            print('IDK lol, ', end='')
            if val_ <= 0.5:
                print('kinda cat-y', end='')
            else:
                print('kinda dog-y', end='')
        elif 0.7 <= val_ <= 1.0:
            print('Probably a dog', end='')
        print('({})'.format(str(val_)))
        return val_


class Counter:
    def __init__(self):
        self.c = 0


class ViewCountCallback(Callback):
    def __init__(self, counter):
        super().__init__()
        self.__counter__ = counter

    def on_epoch_end(self, epoch, logs=None):
        self.__counter__.c += self.params['steps']


class ModelCheckpointCounter(ModelCheckpoint):
    def __init__(self, filename, working_dir, counter):
        super().__init__('placeholder')
        self.filename = filename
        self.counter = counter
        self.working_dir = working_dir

    def refresh_filepath(self):
        self.filepath = self.working_dir + str(self.counter.c) + self.filename

    def on_epoch_end(self, epoch, logs=None):
        self.refresh_filepath()
        super().on_epoch_end(epoch, logs)


def train_checkpoints(model, view_counter, batch_size=100, steps_per_epoch=500, epochs=20, ers=False):
    callback_list = []
    if ers:
        ers = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=1)
        callback_list.append(ers)

    vcc = ViewCountCallback(view_counter)
    callback_list.append(vcc)
    mcc = ModelCheckpointCounter('_vloss{val_loss:.3f}_vacc{val_acc:.3f}_loss{loss:.3f}_acc{acc:.3f}.h5',
                                 curdir, view_counter)
    callback_list.append(mcc)
    return train_model(model, batch_size, steps_per_epoch, epochs, callbacks=callback_list)


def set_lr(model, lr):
    K.set_value(model.optimizer.lr, lr)

model = models.load_model()

# cl = CatDogClassifier(models.load_model('checkpoints/2018-12-11_18-16-52/'
#                                        '4000_vloss0.197_vacc0.941_loss0.061_acc0.977.h5'))

# BATCH_SIZE = 100
# iteration_counter = Counter()
# model = new_model()
# model.summary()
# input()
# save_model(model, curdir + 'arch.json')
# h = train_checkpoints(model, iteration_counter, BATCH_SIZE, 500, 20)
# # train_model(model, 100, 1000, 1)

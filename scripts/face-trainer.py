from keras.layers import Input, Lambda, Dense, Flatten, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
# re-size all the images to this
from tensorboard.program import TensorBoard
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop


def Fc(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)
    return top_model


IMAGE_SIZE = [224, 224]
batch_size = 32
train_path = '../Datasets/face/train'
valid_path = '../Datasets/face/val'
# useful for getting number of classes
folders = glob('../Datasets/face/train/*')

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False

# our layers - you can add more if you want
FC_Head = Fc(vgg, len(folders))

# x = Dense(1000, activation='relu')(x)
# create a model object
model = Model(inputs=vgg.input, outputs=FC_Head)
# view the structure of the model
model.summary()

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=45,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('../Datasets/face/train',
                                                 target_size=IMAGE_SIZE,
                                                 batch_size=batch_size,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('../Datasets/face/val',
                                            target_size=IMAGE_SIZE,
                                            batch_size=batch_size,
                                            class_mode='categorical')

'''r=model.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 2000)'''

# fit the model
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, restore_best_weights=True)
check_point = ModelCheckpoint(r"..\models\{}".format('face_low_val_loss.h5'), monitor='val_loss', mode='min',
                              save_best_only=True, verbose=1)

callbacks = [early_stop, check_point]

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=50,
    validation_data=test_set,
    validation_steps=len(test_set),
    callbacks=callbacks)

# model.save(r"..\models\{}".format('face_model_finale.h5'))

# loss
"""plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')"""

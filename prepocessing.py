import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


base_dir = "E:/Kuliah/Projek/Visi Komputer/visicomputer/eye-dataset"
train_val_dir = os.path.join(base_dir)

print(os.listdir(train_val_dir))

train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.3)
    
validation_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.3)

train_generator = train_datagen.flow_from_directory(
    train_val_dir,
    target_size=(150, 150),
    classes = ['close_look','forward_look','left_look','right_look'],
    class_mode='categorical',
    subset='training')

validation_generator = validation_datagen.flow_from_directory(
    train_val_dir,
    target_size=(150, 150),
    classes = ['close_look','forward_look','left_look','right_look'],
    class_mode='categorical',
    subset='validation')

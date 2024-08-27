import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import keras
from keras import layers
from keras.optimizers import Adam
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

f = open("flower_images copy/flower_labels.csv")
names = f.readlines()
data = []
labels = []
img_rows = 32
img_cols = 32

for i in range(1, len(names), 1):
    names[i] = names[i].strip('\n')
    l = names[i].split(',')
    if (l[1] == "0"):
        labels.append(0)
        img = Image.open("flower_images copy/" + l[0])
        img = img.resize((img_rows, img_cols))
        img_array = np.array(img, dtype=np.float32)
        data.append(img_array)
    elif (l[1] == "5"):
        labels.append(1)
        img = Image.open("flower_images copy/" + l[0])
        img = img.resize((img_rows, img_cols))
        img_array = np.array(img, dtype=np.float32)
        data.append(img_array)

data = np.array(data)
labels = np.array(labels)
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, stratify=labels)
print(X_train.shape)
print(y_train.shape)
print(data.shape)
print(labels.shape)
num_classes = 1
input_shape = (32, 32, 4)

def make_model (input_shape, num_classes):
    inputs=keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(16, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    return keras.Model(inputs, outputs)

newopt = Adam(learning_rate=0.001)
model = make_model(input_shape, num_classes=1)
model.compile(optimizer=newopt, loss='binary_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(
    datagen.flow(X_train, y_train, batch_size=2),
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[lr_scheduler, early_stopping]
)

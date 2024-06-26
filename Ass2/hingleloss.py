
import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import cv2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

faces={}
target_size = (218, 178)
image_dir_map ={"faces":['/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba'],
                  "nfaces":['/kaggle/input/natural-images/natural_images/airplane',
                            '/kaggle/input/natural-images/natural_images/car',
                           '/kaggle/input/natural-images/natural_images/cat',
                           '/kaggle/input/natural-images/natural_images/dog',
                           '/kaggle/input/natural-images/natural_images/flower',
                           '/kaggle/input/natural-images/natural_images/fruit',
                           '/kaggle/input/natural-images/natural_images/motorbike']}
for key,val in image_dir_map.items():
    for image_directory in val:
        for root, _, filenames in os.walk(image_directory):
            if key =='faces':
                img_len=1200
                face=1
            else :
                img_len=150
                face=-1
            i=0
            for filename in filenames:
                i+=1
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                current_size = image.shape[:2]
                if current_size[0] != target_size[0] or current_size[1] != target_size[1]:
                    image = cv2.resize(image, (target_size[1], target_size[0]),interpolation = cv2.INTER_AREA)
#                 else :
#                     vertical_pad = target_size[0] - current_size[0]
#                     horizontal_pad = target_size[1] - current_size[1]
#                     top_pad = vertical_pad // 2
#                     bottom_pad = vertical_pad - top_pad
#                     left_pad = horizontal_pad // 2
#                     right_pad = horizontal_pad - left_pad
#                     image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
#              
                if image.shape[:2][0]==218 and image.shape[:2][1]==178:    
                    image_array = np.array(image)
                    faces[filename]=[image_array , face]
                else:
                    print(f'Not correct size {image_path}')
                if i==img_len:
                    break


model = keras.Sequential([
    layers.Input(shape=(218, 178, 3)),
    layers.Flatten(),
    layers.Dense(512, activation='relu', kernel_regularizer=l2(0.01)),  # Layer 1 with L2 regularization
    layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),  # Layer 2 with L2 regularization
    layers.Dense(100, activation='relu', kernel_regularizer=l2(0.01)),  # Layer 3 with L2 regularization
    layers.Dense(25, activation='relu', kernel_regularizer=l2(0.01)),   # Layer 4 with L2 regularization
    layers.Dense(10, activation='relu', kernel_regularizer=l2(0.01)),   # Layer 5 with L2 regularization
    layers.Dense(1, activation='linear')  # Output layer with linear activation
])

early_stopping = EarlyStopping(
    monitor='val_loss',   # Monitor validation loss
    patience=5,           # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore model weights from the epoch with the best validation loss
)

def hinge_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32) 
    return tf.maximum(0.0, 1.0 - y_true * y_pred)

faces_list = list(faces.items())
random.shuffle(faces_list)
faces = dict(faces_list)

model.compile(optimizer='adam', loss=hinge_loss,  metrics=[tf.keras.metrics.Hinge()])

X = [item[0] for item in faces.values()]
y = [item[1] for item in faces.values()]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

epochs = 50
batch_sz=32
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_sz,validation_data=(X_val, y_val), callbacks=[early_stopping])
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(train_loss) + 1)
plt.figure(figsize=(12, 6))
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

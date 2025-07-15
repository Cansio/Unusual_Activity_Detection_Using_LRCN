# Import required libraries
import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
import gc
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision

# Load dataset
data = np.load(r"300_lv_v_nv_dataset.npz")
features = data['features']
labels = data['labels']
print("Data loaded successfully.")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from numba import cuda
cuda.select_device(0)
cuda.close()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
        )
        print("Memory growth set for GPU")
    except RuntimeError as e:
        print(e)

# Enable mixed precision
mixed_precision.set_global_policy('mixed_float16')
print("GPU Configuration Applied Successfully!")

seed_constant = 25
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["LiftVandalism","Violence","NonViolence"]

# Preprocess labels
one_hot_encoded_labels = to_categorical(labels)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    features, one_hot_encoded_labels, test_size=0.25, shuffle=True, random_state=seed_constant
)

# Model architecture
def create_LRCN_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
                              input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.001))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.001))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(TimeDistributed(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(0.001))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(64)))

    model.add(Dense(len(CLASSES_LIST), activation='softmax'))

    model.summary()
    return model

# Create model
LRCN_model = create_LRCN_model()
print("Model created successfully!")

# Callbacks
early_stopping_callback = EarlyStopping(
    monitor='val_loss', patience=15, mode='min', restore_best_weights=True
)

# Optimizer with custom learning rate
custom_learning_rate = 1e-4
optimizer = Adam(learning_rate=custom_learning_rate)

# Compile model
LRCN_model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=["accuracy"]
)

# Train model
LRCN_model_training_history = LRCN_model.fit(
    x=X_train,
    y=y_train,
    epochs=150,
    batch_size=8,
    shuffle=True,
    validation_split=0.2,
    callbacks=[early_stopping_callback]
)

# Evaluate model
model_evaluation_history_LRCN = LRCN_model.evaluate(X_val, y_val)
model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history_LRCN

# Save model with timestamp
date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

model_file_name = f'liftv_vio_nonvio_model2{current_date_time_string}___Loss_{model_evaluation_loss:.4f}___Accuracy_{model_evaluation_accuracy:.4f}.h5'
LRCN_model.save(model_file_name)

# Accuracy plot
plt.plot(LRCN_model_training_history.history['accuracy'], label='Train Accuracy')
plt.plot(LRCN_model_training_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')
plt.show()

# Loss plot
plt.plot(LRCN_model_training_history.history['loss'], label='Train Loss')
plt.plot(LRCN_model_training_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.show()

# Predictions
predictions = LRCN_model.predict(X_val)
predicted_classes = predictions.argmax(axis=1)
true_classes = y_val.argmax(axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

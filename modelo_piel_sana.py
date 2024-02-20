# import cv2
import tensorflow as tf
import os
from time import sleep
from keras import Model
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.layers import BatchNormalization, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras import backend as k
import numpy as np
from matplotlib import pyplot as plt
# import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
classes = [] # Acne, Psoriasis, Alopecia, Melanoma, Eczema, Poison Ivy, Lupus, Light Diseases, Nail fungus
epochs_recorridas = 0
acc_final = 0
val_acc_final = 0


class callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > logs.get('val_accuracy')+0.15):
            print("\nSe observa overfit, deteniendo entrenamiento y guardando el modelo")
            acc_final = logs.get('accuracy')
            val_acc_final = logs.get('val_accuracy')
            self.model.stop_training = True

callback = callback()

tf.config.list_logical_devices('CPU')
for carpeta in os.listdir("dataset_pielSana_resized/train"):
    classes.append(carpeta)

output_size = len(classes)
epoch = 20

print("Iniciando")

print("Cargando datasets")
data_train = tf.keras.utils.image_dataset_from_directory('dataset_pielSana_resized/train',
    labels='inferred',
    label_mode='binary',
    class_names=classes,
    batch_size=32)
data_test  = tf.keras.utils.image_dataset_from_directory('dataset_pielSana_resized/test',
    labels='inferred',
    label_mode='binary',
    class_names=classes,
    batch_size=32)

train_label = np.concatenate([y for x, y in data_train], axis=0)
data_train_iterator = data_train.as_numpy_iterator()
data_test_iterator  = data_test.as_numpy_iterator()
batch = data_train_iterator.next()

# fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img.astype(int))
#     ax[idx].title.set_text(batch[1][idx])
#plt.show()

train_size = int(len(data_train)*.8)
val_size = int(len(data_train)*.2)
test_size  = int(len(data_test))

train = data_train.take(train_size)
val = data_train.skip(train_size).take(val_size)
test = data_test.take(test_size)

print("Creando modelo")

res = ResNet50(weights='imagenet',include_top=False,input_shape=(256, 256, 3))
res.trainable = False

x = res.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(64, activation ='relu')(x)
x = BatchNormalization()(x)

x = Dense(128, activation ='relu')(x)
x = BatchNormalization()(x)

x = Dense(128, activation ='relu')(x)
x = BatchNormalization()(x)

x = Dense(1, activation ='sigmoid')(x)
model = Model(res.input, x)

model.compile('adam', loss="binary_crossentropy", metrics=['accuracy'])
model.summary()
print("Entrenando modelo")
print(model.count_params())
logdir='logs'
checkpoint_path = "logs/ckpt.20e"
checkpoint_dir = os.path.dirname(logdir)

# hist = model.fit(train, epochs=epoch, validation_data=val, callbacks=[callback])
hist = model.fit(train, epochs=epoch, validation_data=val)
model.evaluate(data_test)
carpeta = "logs/modelo_PielSana(ResNetV4)_("+str(hist.history['accuracy'][len(hist.history['accuracy'])-1])+"_"+str(hist.history['val_accuracy'][len(hist.history['accuracy'])-1])+"acc(T_V)-"+str(epoch)+"E)"

model.save(carpeta)
model.save("PielSana(ResNetV5).h5")

fig, (ax1,ax2) = plt.subplots(1,2)
#plot accuracy vs epoch
ax1.plot(hist.history['accuracy'])
ax1.plot(hist.history['val_accuracy'])
ax1.set_title('Model accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Test'], loc='upper left')
ax1.set_xlim(0,20)
ax1.grid(True)
# plt.show()
# plt.savefig(carpeta+'/Accuracy.png')

# Plot loss values vs epoch
ax2.plot(hist.history['loss'])
ax2.plot(hist.history['val_loss'])
ax2.set_title('Model loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Test'], loc='upper right')
ax2.set_xlim(0,20)
ax2.grid(True)
fig.savefig('Metrics ResNet50_v4.png')
plt.show()

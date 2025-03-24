import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, MobileNet, EfficientNetB0, MobileNetV3Small
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications.imagenet_utils import preprocess_input

from augmentdata import data_augmentation

MODEL = 'mobilenet'

# ---------------------- PARAMETERS ---------------------- #
# Parametros (TODO: cargar configuracion de un fichero)
IMG_HEIGHT = 64 # 128 # cuadrado 128x128?
IMG_WIDTH = 384 #512
CHANNELS = 3
#NUM_CLASSES = 11 # obtained from IamgeDataGenerator
BATCH_SIZE = 32

if MODEL=='mobilenet':
    INITIAL_LR = 5e-4 # 1st training
else:
    INITIAL_LR = 1e-3 # 1st training
    
EPOCHS1 = 15

if MODEL=='mobilenet':
    UNFREEZE = 50 # number of layers to unfreeze
    FT_LR = 1e-4 # fine-tune
else:
    UNFREEZE = 40 # number of layers to unfreeze
    FT_LR = 1e-4 # fine-tune
    
EPOCHS2 = 150

PATIENCE = 15

# Datos
TRAIN_DIR = "/tfm-external/birdnet-output_train/imgs/"  # Ruta de la carpeta con imágenes organizadas por categorías
VALID_DIR = "/tfm-external/birdnet-output_val/imgs/"  # Ruta de la carpeta con imágenes organizadas por categorías
TEST_DIR = "/tfm-external/birdnet-output_test/imgs/"  # Ruta de la carpeta con imágenes organizadas por categorías


# ---------------------- DATA LOADING AND PROCESSING. DATASET ---------------------- #
def to_rgb(image):
    return np.stack([image]*3, axis=-1)

if MODEL=='mobilenet':
    rescaling= 1.0/255.0
    preprocessing = None
else:
    rescaling= None
    preprocessing = preprocess_input
    
train_datagen = ImageDataGenerator(
    rescale= rescaling, #1.0 / 255.0,
    preprocessing_function=data_augmentation,
    #preprocessing_function=preprocessing,
    #rotation_range=15,
    width_shift_range=0.1,
    #height_shift_range=0.1,
    )

valid_datagen = ImageDataGenerator(rescale=rescaling, preprocessing_function=preprocessing) 
test_datagen = ImageDataGenerator(rescale=rescaling, preprocessing_function=preprocessing)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH),  batch_size=BATCH_SIZE, class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical')

LABELS = list(train_generator.class_indices.keys())
print(f"Target categories {LABELS}")
NUM_CLASSES = len(LABELS)

# Gestionar desbalanceo
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))


labels, counts = np.unique(train_generator.classes, return_counts=True)
plt.bar(labels, counts)
plt.xlabel("Clases")
plt.ylabel("Cantidad de imágenes")
plt.title("Distribución de clases en el conjunto de entrenamiento")
plt.show()
plt.savefig('DistribucionClases.png')
#sys.exit(0)

# Check dataset
'''image, label = train_generator.next()
print("Image shape:", image.shape)
print("Label shape:", label.shape)
print(label)
print(image.max(), image.min())
#print(image)
sys.exit(0)'''



# ---------------------- DEFINE MODEL ---------------------- #
#base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
if MODEL=='mobilenet':
    base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)) 
else:
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Añadimos ultimas capas 'head'
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),  
    layers.Dense(NUM_CLASSES, activation='softmax')
])

#print(model.input)
#sys.exit(0)

# ---------------------- TRAIN MODEL ---------------------- #

# (1) Entrenar solo el 'head'
# ----------------------------------------------------------------------
base_model.trainable = False  # no entrenar inicialmente el modelo 

#initial_lr = 5e-4
model.compile(optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=PATIENCE, restore_best_weights=True
)

model.summary()

history = model.fit(
    train_generator,
    #steps_per_epoch=100, # cambiado
    validation_data=valid_generator,
    epochs=EPOCHS1,
    class_weight=class_weights,  # desalanceo de clases
    callbacks=[early_stopping],
    verbose=1
)

print((model.layers[0]).layers[-2].get_weights()[0])


# (2) Entrenar mas capas
# ----------------------------------------------------------------------
for layer in model.layers:
    layer.trainable = True
    
for layer in (model.layers[0]).layers[:-UNFREEZE]:#[:100]: 
    layer.trainable = False
        
model.summary()

print((model.layers[0]).layers[-2].get_weights()[0])

#fine_tune_lr = 1e-4
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=FT_LR, decay_steps=1000, decay_rate=0.96, staircase=True 
)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),#FT_LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

'''for layer in model.layers[0].layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False'''

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  factor=0.5, patience=int(PATIENCE/2), min_lr=1e-6, verbose=1         
)

history_fine = model.fit(
    train_generator,
    validation_data=valid_generator,
    #steps_per_epoch=100, # cambiado
    initial_epoch = EPOCHS1,
    epochs= EPOCHS1+EPOCHS2, #150,
    class_weight=class_weights,
    callbacks=[early_stopping],#, reduce_lr],
    verbose=1
)

print((model.layers[0]).layers[-2].get_weights()[0])


'''
# No ha sido muy efectivo:
# (3) Entrenar mas capas
# ----------------------------------------------------------------------
for layer in model.layers:
    layer.trainable = True
    
for layer in (model.layers[0]).layers[:-UNFREEZE-20]:
    layer.trainable = False
        
model.summary()

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=FT_LR/10, decay_steps=1000, decay_rate=0.96, staircase=True 
)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),#FT_LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine2 = model.fit(
    train_generator,
    validation_data=valid_generator,
    initial_epoch = EPOCHS1+EPOCHS2,
    epochs= EPOCHS1+EPOCHS2+50, #150,
    class_weight=class_weights,
    callbacks=[early_stopping],#, reduce_lr],
    verbose=1
)'''


def plot_history_1(acc,val_acc,loss,val_loss, namefig='fig1.png'):
  plt.figure(figsize=(12, 12))
  plt.subplot(2, 1, 1)
  N = len(acc) # total number of epochs
  plt.plot(np.arange(1,N+1), acc, '-o', label= "Training Accuracy")
  plt.plot(np.arange(1,N+1), val_acc, '-o', label= "Validation Accuracy")
  plt.legend(loc='lower right')
  plt.ylabel('Accuracy')
  #plt.ylim([min(plt.ylim()),1])
  plt.title('Training and Validation Accuracy')

  plt.text(1, acc[-1], "Training Accuracy: {:.2f}".format(acc[-1]))
  plt.text(1, val_acc[-1], "Val Accuracy: {:.2f}".format(val_acc[-1]))

  plt.subplot(2, 1, 2)
  plt.plot(np.arange(1,N+1), loss, '-s', label= "Training Loss")
  plt.plot(np.arange(1,N+1), val_loss, '-s', label= "Validation Loss")
  plt.legend(loc='upper right')
  plt.ylabel('Cross Entropy')
  #plt.ylim([0,1.0])
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.text(1, loss[-1], "Training Loss: {:.2f}".format(loss[-1]))
  plt.text(1, val_loss[-1], "Val Loss: {:.2f}".format(val_loss[-1]))

  plt.show()
  plt.savefig(namefig, bbox_inches='tight'); print( 'Saved as ' + namefig )

plot_history_1( list(history.history['accuracy']) + list(history_fine.history['accuracy']), 
	list(history.history['val_accuracy']) + (history_fine.history['val_accuracy']),
	list(history.history['loss']) + list(history_fine.history['loss']),
	list(history.history['val_loss']) + list(history_fine.history['val_loss']) )
	


# ---------------------- TEST MODEL ---------------------- #
# Evaluacion en conjunto de test
# ----------------------------------------------------------------------
#test_loss, test_acc = model.evaluate(test_generator, verbose=1)
#print(f"Test accuracy: {test_acc:.4f}")

model.save("mobilenet_spectrogram.h5")

test_generator = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)
    
true_classes = test_generator.classes[test_generator.index_array].squeeze()  # esto debe ir antes que predict() 
predIdxs = model.predict_generator(test_generator,steps=None) #,steps=1 )
predIdxs = np.argmax(predIdxs, axis=1)
#print(true_classes)
#print(predIdxs)

from sklearn.metrics import classification_report, confusion_matrix
print('Confusion Matrix')
cm = confusion_matrix(true_classes, predIdxs) #, labels=LABELS)
print(cm) 
print('Accuracy {:.2f}%'.format( 100*sum( (predIdxs.squeeze()==true_classes))/ true_classes.shape[0] ) ) 

from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
disp.plot()
plt.show()
plt.savefig('confusion_matrix.png')

print(classification_report(true_classes, predIdxs, target_names=LABELS))

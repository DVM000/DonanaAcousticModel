import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNet
import sys
import json
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import tqdm
import random
from tensorflow.keras.models import load_model

from augmentdata import data_augmentation

from  util import bcolors, check_GPU, plot_confusion_matrix, plot_classification_report, plot_history_1, calculate_metrics, plot_ROC

#https://pyimagesearch.com/2021/06/28/data-augmentation-with-tf-data-and-tensorflow/


# ---------------------- PARAMETERS ---------------------- #

TRAIN_IMAGE_DIR = "./tfm-external/birdnet-output_train/imgs/"  # Ruta con subcarpetas de imagenes por categorias
TRAIN_NPY_DIR = "./tfm-external/birdnet-output_train//npy/"  # Ruta a archivos .npy de soft labels

VAL_IMAGE_DIR = "./tfm-external/birdnet-output_val/imgs/"  
VAL_NPY_DIR = "./tfm-external/birdnet-output_val/npy/"  

TEST_IMAGE_DIR = "./tfm-external/birdnet-output_test/imgs/"  
TEST_NPY_DIR = "./tfm-external/birdnet-output_test/npy/"  

TRAIN_IMAGE_DIR = "./tfm-external/less_classes/train/imgs/" #  Falco + Larus. 
VAL_IMAGE_DIR = "./tfm-external/less_classes/val/imgs/"
TEST_IMAGE_DIR = "./tfm-external/less_classes/test/imgs/"
TRAIN_NPY_DIR = "./tfm-external/less_classes/train/npy/"
VAL_NPY_DIR = "./tfm-external/less_classes/val/npy/"
TEST_NPY_DIR = "./tfm-external/less_classes/test/npy/"

# ABclasses
'''
TRAIN_IMAGE_DIR = "./AUDIOSTFM/ABclasses/train/" 
VAL_IMAGE_DIR = "./AUDIOSTFM/ABclasses/val/"
TEST_IMAGE_DIR = "./AUDIOSTFM/ABclasses/test/"
TRAIN_NPY_DIR = "./AUDIOSTFM/train_npy/npy/"
VAL_NPY_DIR = "./AUDIOSTFM/val_npy/npy/"
TEST_NPY_DIR = ""
'''

# ALL
TRAIN_IMAGE_DIR = "./AUDIOSTFM/train_imgs/" # todas las especies
VAL_IMAGE_DIR = "./AUDIOSTFM/val_imgs/"
TEST_IMAGE_DIR = "./AUDIOSTFM/test_imgs/"
TRAIN_NPY_DIR = "./AUDIOSTFM/train_npy/npy/"
VAL_NPY_DIR = "./AUDIOSTFM/val_npy/npy/"
TEST_NPY_DIR = "./AUDIOSTFM/test_npy/npy/"

'''
TRAIN_IMAGE_DIR = "/dataset/AUDIOS/TFM/ABclasses/train/" 
VAL_IMAGE_DIR = "/dataset/AUDIOS/TFM/ABclasses/val/"
TEST_IMAGE_DIR = "/dataset/AUDIOS/TFM/ABclasses/test/"
'''

LOADmodel = False
MODEL_PATH = "../Models/mobilenet-224-337wi.h5"

exp_sufix = '-prueba' # sufix for plot figures generated in this experiment

MAX_PER_CLASS = 1000 # maximum data to take of each category
MIN_PER_CLASS = 50 # minimum data to take of each category

IMG_HEIGHT = 224 #128
IMG_WIDTH = IMG_HEIGHT
CHANNELS = 3
#NUM_CLASSES = 5 # obtained from Dataset
BATCH_SIZE = 32

rescaling = 1.0 / 255.0  # Normalizacion

INITIAL_LR = 5e-4 # 1st training
EPOCHS1 = 10 

UNFREEZE = -1 # number of layers to unfreeze
FT_LR = 5e-4  # fine-tune
EPOCHS2 = 60 #150

if LOADmodel:
    FT_LR = 5e-3 
    EPOCHS1 = 3
    EPOCHS2 = 0
    MIN_PER_CLASS = 1
    MAX_PER_CLASS = 500
    
    SELECTED_SPECIES_FILE = f'../Models/species-list-337.txt'
    with open(SELECTED_SPECIES_FILE, "r") as f:
        LABELS_ORIG = [line.strip() for line in f] 
    
PATIENCE = 10 #15

alpha = 1  #  0.5  # Peso de las hard labels en la mezcla (1-alpha = peso de soft labels)
temperature = 2.0  # Parámetro para suavizar soft labels

# Si alpha=1, no aplicamos distillation, no necesitamos las soft labels ni la funcion KL divergence y el entrenamiento es mas rapido

DICT_BIRDNET = f'../BirdNet/birdnet_idx.json' # Diccionario con key: species, valor: indice en output BirdNET +1    

# ---------------------- CHECK GPU AVAILABLE ---------------------- #
gpus = check_GPU()

# ---------------------- DATA LOADING ---------------------- #
def load_data(image_path, npy_path, MAX_PER_CLASS, MIN_PER_CLASS, category_list=[], allow_missing_npy=True):
    """
    Carga imagenes y soft labels.

    image_path: Ruta a imagenes en subcarpetas.
    npy_path: Ruta a archivos .npy.
    category_list: Lista deseada de categorias (subdirectorios a buscar). Si no se especifica, se toma de subdirectorios.
    allow_missing_npy: Si es True, permite cargar imágenes aunque no exista su correspondiente archivo .npy.

    Devuelve listas de rutas de imagenes y sus etiquetas.
    """
    image_files = []
    soft_labels = []
    hard_labels = []
    category_list_min = [] # lista de categorias que superan el minimo de datos
    
    # Recorrer categorías (subcarpetas)
    if len(category_list)<1:
        category_list = sorted(os.listdir(image_path)) # subcarpetas si no se especifica lista
    '''    MIN_ = MIN_PER_CLASS 
    else:
        MIN_ = 0  # si se especifica lista, no usamos minimo'''
        
    ii = -1
    for i,category in enumerate(category_list):
        
        category_img_path = os.path.join(image_path, category)
        category_npy_path = npy_path #os.path.join(npy_path, category)

        if not os.path.isdir(category_img_path):
            continue  # Saltar archivos que no sean carpetas

        image_list = os.listdir(category_img_path)
        
        if len(image_list) < MIN_PER_CLASS: 
            print(f"Skipping category ({i}) {category}: only {len(os.listdir(category_img_path))} files")
            continue
        
        category_list_min.append(category) # Agregar categoría a la lista si ha sido añadida
        ii = ii + 1 # solo si la categoria ha sido añadida  
        
        # Seleccionar hasta MAX_PER_CLASS imágenes de manera aleatoria
        selected_images = random.sample(image_list, min(MAX_PER_CLASS, len(image_list)))
            
        # Recorrer las imágenes dentro de la categoría
        for img_file in selected_images: # os.listdir(category_img_path)[:MAX_PER_CLASS]:
            img_full_path = os.path.join(category_img_path, img_file)
            npy_full_path = os.path.join(category_npy_path, img_file.replace(".png", ".npy"))
            #print(img_full_path)
            #print(npy_full_path)
            
            if os.path.exists(npy_full_path):  # Asegurar que exista la predicción
                image_files.append(img_full_path)
                soft_labels.append(np.load(npy_full_path))  # Cargar soft label desde .npy
                hard_labels.append(ii)
                #print(img_full_path, i)
            elif allow_missing_npy:
                print(f'no npy for {img_full_path}')
                image_files.append(img_full_path)
                soft_labels.append(1e-3*np.ones(6522))  # Agregar array constante de valores pequeños si falta el npy
                hard_labels.append(ii)
                     
        print(f"({i}) {category}: {min(len(os.listdir(category_img_path)[:MAX_PER_CLASS]), len(os.listdir(category_npy_path)[:MAX_PER_CLASS]))} files.")
                
    image_files = tf.constant(image_files, dtype=tf.string)
    #soft_labels = tf.convert_to_tensor(soft_labels, dtype=tf.float32)
    
    print(bcolors.OKCYAN+ f"Found {len(image_files)} images belonging to {len(np.unique(hard_labels))} classes." +bcolors.ENDC)

    return image_files, np.array(soft_labels), hard_labels, category_list_min


def load_data_nonpy(image_path, MAX_PER_CLASS, MIN_PER_CLASS, category_list=[]):
    """
    Carga imágenes y soft labels. A diferencia de la funcion anterior, esta NO busca ficheros .npy (puede ser lento si hay muchos)

    image_path: Ruta a imagenes en subcarpetas.
    category_list: Lista deseada de categorias (subdirectorios a buscar). Si no se especifica, se toma de subdirectorios.

    Devuelve listas de rutas de imágenes y sus etiquetas.
    """
    image_files = []
    hard_labels = []
    category_list_min = [] # lista de categorias que superan el minimo de datos
    
    # Recorrer categorías (subcarpetas)
    if len(category_list)<1:
        category_list = sorted(os.listdir(image_path)) # subcarpetas si no se especifica lista
    '''    MIN_ = MIN_PER_CLASS 
    else:
        MIN_ = 0  # si se especifica lista, no usamos minimo'''
        
    ii = -1
    for i,category in enumerate(category_list):
        
        category_img_path = os.path.join(image_path, category)

        if not os.path.isdir(category_img_path):
            continue  # Saltar archivos que no sean carpetas

        image_list = os.listdir(category_img_path)

        if len(image_list) < MIN_PER_CLASS: 
            print(f"Skipping category ({i}) {category}: only {len(os.listdir(category_img_path))} files")
            continue
        
        category_list_min.append(category)
        ii = ii + 1 # solo si la categoria ha sido añadida  
        
        # Seleccionar hasta MAX_PER_CLASS imágenes de manera aleatoria
        selected_images = random.sample(image_list, min(MAX_PER_CLASS, len(image_list)))
            
        # Recorrer las imágenes dentro de la categoría
        for img_file in os.listdir(category_img_path)[:MAX_PER_CLASS]:
            img_full_path = os.path.join(category_img_path, img_file)
            image_files.append(img_full_path)
            hard_labels.append(ii)
                
        print(f"({i}) {category}: {len(os.listdir(category_img_path)[:MAX_PER_CLASS])} files.")
        
    image_files = tf.constant(image_files, dtype=tf.string)
    #soft_labels = tf.convert_to_tensor(soft_labels, dtype=tf.float32)
               
    print(bcolors.OKCYAN+ f"Found {len(image_files)} images belonging to {len(np.unique(hard_labels))} classes." +bcolors.ENDC)

    return image_files, hard_labels, category_list_min
    
# ---------------------- DATA PROCESSING ---------------------- #
def load_and_preprocess_image(image_path, label, augment=False):
    """
    Carga una imagen y la preprocesa para el entrenamiento.
    
    Parametros:
    - image_path: ruta de la imagen.
    - label: etiqueta.
    - augment: aplicar o no data augmentation.
    """
    if isinstance(image_path, tf.Tensor):
        image_path = image_path.decode("utf-8")  # Convertir a string
    image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))  # Cargar imagen
    image = img_to_array(image) * rescaling  # Convertir a array y normalizar
    
    if augment: # solo para training
        image = data_augmentation(image) #  Add data augmentation
        
    image = tf.cast(image, tf.float32)  # Asegurar float32 después de la data augmentation
    
    '''image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = img_to_array(image) * rescaling  # Normalización'''
    
    # Pasar a tensor
    label = tf.convert_to_tensor(label, dtype=tf.float32)
    #label.set_shape([num_classes])  # Establecer la forma 
    return image, label

def parse_function(image_path, label, augment=False):
    """
    Función de preprocesamiento usando `tf.numpy_function`.
    """
    # Para tf.data.Dataset, necesitamos envolver las funciones de numpy en un tf.Tensor
    image, label = tf.numpy_function(func=load_and_preprocess_image, 
                                     inp=[image_path, label, augment], 
                                     Tout=(tf.float32, tf.float32))

    image.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])  # Definir tamaño de entrada
    #label.set_shape([soft_labels.shape[1]])  
    #num_classes_total = soft_labels.shape[1] + hard_labels.shape[1] 
    #label.set_shape([num_classes_total])  # Definir número de clases *2

    return image, label
    
# Versiones separadas con/sin augmentation
def parse_function_train(image_path, label):
    return parse_function(image_path, label, augment=True)

def parse_function_eval(image_path, label):
    return parse_function(image_path, label, augment=False)
    
# ---------------------- DATASET ---------------------- #
# Cargar datos
print(f"Loading data...")

if alpha<1 and not LOADmodel:
    train_image_files, train_soft_labels, train_hard_labels, LABELS = load_data(TRAIN_IMAGE_DIR, TRAIN_NPY_DIR, MAX_PER_CLASS, MIN_PER_CLASS)
    val_image_files, val_soft_labels, val_hard_labels, _ = load_data(VAL_IMAGE_DIR, VAL_NPY_DIR, int(MAX_PER_CLASS/2), 0, LABELS)
    print(train_image_files[:2])
elif not LOADmodel:
    train_image_files, train_hard_labels, LABELS = load_data_nonpy(TRAIN_IMAGE_DIR, MAX_PER_CLASS, MIN_PER_CLASS)
    val_image_files, val_hard_labels, _ = load_data_nonpy(VAL_IMAGE_DIR, int(MAX_PER_CLASS/2), 0, LABELS)

if LOADmodel: # follow species-list
    train_image_files, train_hard_labels, LABELS = load_data_nonpy(TRAIN_IMAGE_DIR, MAX_PER_CLASS, MIN_PER_CLASS, LABELS_ORIG)
    val_image_files, val_hard_labels, _ = load_data_nonpy(VAL_IMAGE_DIR, int(MAX_PER_CLASS/2), 0, LABELS_ORIG)
               
print(f"Target categories {LABELS}")
NUM_CLASSES = len(LABELS)

# Gestionar desbalanceo
'''
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_hard_labels),
    y=train_hard_labels
)
class_weights = dict(enumerate(class_weights))
'''

labels, counts = np.unique(train_hard_labels, return_counts=True)
plt.bar(labels, counts)
plt.xlabel("Clases")
plt.ylabel("Cantidad de imágenes")
plt.title("Distribución de clases en el conjunto de entrenamiento")
plt.show()
plt.savefig(f'DistribucionClases{exp_sufix}.png')
print(bcolors.OKCYAN+'Saved as ' + f'DistribucionClases{exp_sufix}.png' +bcolors.ENDC)
#sys.exit(0)


# ---------------------- SELECT LABELS ---------------------- #
def selectNsoftlabels_and_gethardlabels(idx, soft_labels, hard_labels):
    ''' - Select only labels in index idx, and apply softmax.
          Convert to tensor.
        - Get hard labels in one-hot codification '''
    soft_labels = soft_labels[:, idx]  # tf.nn.softmax(soft_labels[:, idx], axis=1)  --> softmax se esta aplicando en la funcion de perdidas
    soft_labels = tf.convert_to_tensor(soft_labels, dtype=tf.float32)
    
    hard_labels = np.eye(NUM_CLASSES)[hard_labels]  #  etiquetas one-hot
    return soft_labels, hard_labels

# Diccionario con key: species, valor: indice en output BirdNET +1    
with open(DICT_BIRDNET, 'r') as fp:
    idx_dict = json.load(fp)
#print(idx_dict)

#idx = np.array(list(idx_dict.values()))  - 1
#print('\n ** WARNING: select correct soft labels ** \n\n ')
#idx = [2290-1, 2294-1, 2295-1, 2299-1, 2300-1, 2301-1]

# Select indices from idx_dict and LABELS:
idx = []
for l in LABELS:
    #print(l, idx_dict[l])
    try:
        idx.append(idx_dict[l]-1)
    except:
        idx.append(3927) # Noise
        print(bcolors.WARNING+ f"WARNING: species {l} not found in dictionary" +bcolors.ENDC)
        
print(f"Selected indexes for our categories: {idx}")

# Guardar la lista final de especies usadas para entrenar el modelo
selected_species_file = f'../Models/species-list-model{exp_sufix}.txt'
with open(selected_species_file, 'w') as f:
    for item in LABELS:
        f.write(item + "\n")
print(bcolors.OKCYAN+ f"Saved {selected_species_file} with species list with minimum {MIN_PER_CLASS} data." +bcolors.ENDC)    
#print(np.argmax(train_soft_labels, axis=1))

if alpha<1:
    train_soft_labels, train_hard_labels = selectNsoftlabels_and_gethardlabels(idx, train_soft_labels, train_hard_labels)
    print("Soft labels examples:")
    print(train_soft_labels[:2,:])
    print("... that correspond to:")
    print(np.argmax(train_soft_labels[:2,:], axis=1))
    print("Hard labels examples:")
    print(train_hard_labels[:2,:])
    #sys.exit(0)
    val_soft_labels, val_hard_labels = selectNsoftlabels_and_gethardlabels(idx, val_soft_labels, val_hard_labels)
else:
    train_hard_labels = np.eye(NUM_CLASSES)[train_hard_labels]  #  etiquetas one-hot
    val_hard_labels = np.eye(NUM_CLASSES)[val_hard_labels]  #  etiquetas one-hot
    

if alpha<1:   train_labels = np.concatenate([train_soft_labels, train_hard_labels], axis=1)  # y_true es la concatenacion de soft labels + hard labels
else:        train_labels = train_hard_labels 
train_labels = tf.cast(train_labels, tf.float32)

if alpha<1:   val_labels = np.concatenate([val_soft_labels, val_hard_labels], axis=1) 
else:        val_labels = val_hard_labels 
val_labels = tf.cast(val_labels, tf.float32)
    
# ---------------------- DATASET ---------------------- #
#https://www.tensorflow.org/guide/data?hl=en
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_files, train_labels)) # 
train_dataset = train_dataset.shuffle(buffer_size=len(train_image_files)) # necesario poner esto antes que .batch()
train_dataset = train_dataset.map(parse_function_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
if len(gpus):
    train_dataset = train_dataset.apply(tf.data.experimental.prefetch_to_device('/GPU:0')) # para que use la GPU, por defecto no la estaba usando

# Verificar dataset
'''for image, label in train_dataset.take(10):
    print("Imagen shape:", image.shape)
    print("Etiqueta shape:", label.shape)
    print(label[:5,:])
    #print(image)
    print(np.max(image), np.min(image))
  
#import matplotlib.pyplot as plt
#plt.figure()
#plt.imshow(image[0])
#plt.savefig("Dataset.png")
sys.exit(0)'''
'''
for batch in train_dataset.take(1):
    print(batch[0].dtype, batch[1].dtype)
sys.exit(0)'''

val_dataset = tf.data.Dataset.from_tensor_slices((val_image_files, val_labels)) # 
val_dataset = val_dataset.shuffle(buffer_size=len(val_image_files))  
val_dataset = val_dataset.map(parse_function_eval, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
if len(gpus):
    val_dataset = val_dataset.apply(tf.data.experimental.prefetch_to_device('/GPU:0')) 

# ---------------------- DEFINE MODEL ---------------------- #
base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Añadimos ultimas capas 'head'
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu', name="dense_output_1"),
    layers.Dropout(0.3),  
    layers.Dense(NUM_CLASSES, activation='softmax', name="dense_output_2")
])


print(model.input)
'''try:
    for i, layer in enumerate(model.layers[0].layers):
        print(i, layer.name)
except:
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
sys.exit(0)'''

# ---------------------- DEFINIR PÉRDIDA DE DISTILLATION ---------------------- #
# ---------------------- MEZCLA DE HARD Y SOFT LABELS ---------------------- #

# https://keras.io/examples/vision/knowledge_distillation/

def distillation_loss(y_true, y_pred):
    """
    Funcion de perdidas mezcla de crossentropy y KL Divergence.
    """
    num_classes = y_pred.shape[-1]  # Asegurar que num_classes es correcto
    hard_labels = y_true[:, -num_classes:]
    soft_labels = y_true[:, :num_classes] 
    #print(hard_labels); print(soft_labels); print(y_pred)
    
    # Convertir a float32 
    soft_labels = tf.cast(soft_labels, tf.float32)
    hard_labels = tf.cast(hard_labels, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Asegurar que soft_labels y y_pred sean distribuciones de probabilidad
    soft_labels_softmax = tf.nn.softmax(soft_labels / temperature, axis=-1)
    y_pred_softmax = tf.nn.softmax(y_pred / temperature, axis=-1)  # Para KL Divergence

    # Cálculo de funciones de perdidas
    kl_loss = tf.keras.losses.KLDivergence()(soft_labels_softmax, y_pred_softmax) * temperature**2
    hard_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(hard_labels, y_pred)
    #print("Hard Loss:", hard_loss.numpy(), "KL Loss:", kl_loss.numpy())

    # Mezcla ponderada de perdidas
    total_loss = alpha * hard_loss + (1 - alpha) * kl_loss
    return total_loss

def custom_accuracy(y_true, y_pred):
    """
    Accuracy a partir de hard labels. y_true es la concatenacion de soft labels + hard labels
    """
    num_classes = tf.shape(y_pred)[-1]  # Número de clases 

    # Extraer hard labels (ultimas columnas de y_true)
    hard_labels = y_true[:, -num_classes:]
    true_classes = tf.argmax(hard_labels, axis=-1)

    # Prediccion del modelo
    pred_classes = tf.argmax(y_pred, axis=-1)

    # Comparar y calcular accuracy
    accuracy = tf.cast(tf.equal(true_classes, pred_classes), dtype=tf.float32)
    return tf.keras.backend.mean(accuracy)  # promedio de correctos

'''
print(train_labels[:10,:])  
print(train_labels[:10, -NUM_CLASSES:] )
print(distillation_loss(train_labels[:10,:], train_labels[:10, -NUM_CLASSES:]))
sys.exit(0)  ''' 

if LOADmodel:
    model = load_model(MODEL_PATH, custom_objects={"distillation_loss": distillation_loss, "custom_accuracy": custom_accuracy})
    base_model = model.layers[1] 
    for layer in model.layers[:-2]: # freeze the whole model except for the last dense(+softmax) layer (just one layer)
        layer.trainable = False

  
# ---------------------- TRAIN MODEL ---------------------- #
METRICS = [custom_accuracy] if alpha<1 else ['categorical_accuracy']
LOSS    = distillation_loss  if alpha<1 else 'categorical_crossentropy'

# (1) Entrenar solo el 'head'
# ----------------------------------------------------------------------
base_model.trainable = False  # no entrenar inicialmente el modelo 
model.summary()

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=PATIENCE, restore_best_weights=True
)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
              #loss=distillation_loss,
              ##metrics=['categorical_accuracy'],) 
              #metrics=[custom_accuracy]) #categorical_accuracy
              loss=LOSS, 
              metrics=METRICS)
  
history = model.fit(train_dataset, 
		  validation_data=val_dataset,
		  epochs=EPOCHS1,
		  #class_weight=class_weights,  # desalanceo de clases
                  callbacks=[early_stopping],
		  verbose=1)
		  

# (2) Entrenar mas capas
# ----------------------------------------------------------------------
for layer in model.layers:
    layer.trainable = True
    
try:
    for layer in (model.layers[0]).layers[:-UNFREEZE]:#[:100]:
        layer.trainable = False
except:
    print('Error when unfreezing layers')
            
model.summary()

# Learning rate scheduler with Exponential decay:
try:
    steps_per_epoch = train_dataset.cardinality().numpy()
    if steps_per_epoch == tf.data.INFINITE_CARDINALITY or steps_per_epoch == tf.data.UNKNOWN_CARDINALITY:
        raise ValueError("Unknown Cardinality")
except:
    steps_per_epoch = sum(1 for _ in train_dataset)
    
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=FT_LR, decay_steps= steps_per_epoch * 2, decay_rate=0.96, staircase=True  # Reduce LR each 2 epochs
)

class PrintLearningRate(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        # Convert variable/tensor into float
        lr = tf.keras.backend.get_value(lr)
        print(f"\nEpoch {epoch + 1}: Learning rate is {lr:.6f}")

model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),#FT_LR),
              #loss=distillation_loss,
              #metrics=[custom_accuracy]) #sparse_categorical_accuracy
              loss=LOSS, 
              metrics=METRICS)
    
history_fine = model.fit(train_dataset, 
		  validation_data=val_dataset,
		  initial_epoch = EPOCHS1,
                  epochs= EPOCHS1+EPOCHS2, 
                  #class_weight=class_weights,
                  callbacks=[early_stopping], #, PrintLearningRate()],#, reduce_lr],
		  verbose=1)

if alpha<1 and EPOCHS1 and EPOCHS2:
    plot_history_1( list(history.history['custom_accuracy']) + list(history_fine.history['custom_accuracy']), 
	list(history.history['val_custom_accuracy']) + (history_fine.history['val_custom_accuracy']),
	list(history.history['loss']) + list(history_fine.history['loss']),
	list(history.history['val_loss']) + list(history_fine.history['val_loss']), namefig=f'fig1-{exp_sufix}.png' )
if alpha==1 and EPOCHS1 and EPOCHS2:
    plot_history_1( list(history.history['categorical_accuracy']) + list(history_fine.history['categorical_accuracy']), 
	list(history.history['val_categorical_accuracy']) + (history_fine.history['val_categorical_accuracy']),
	list(history.history['loss']) + list(history_fine.history['loss']),
	list(history.history['val_loss']) + list(history_fine.history['val_loss']), namefig=f'fig1-{exp_sufix}.png' )

# ---------------------- TEST MODEL ---------------------- #
model.save(f"../Models/mobilenet-{exp_sufix}.h5")
print(bcolors.OKCYAN+'Saved as ' + f'mobilenet-{exp_sufix}.h5' +bcolors.ENDC)

#test_image_files, _, test_hard_labels, _ = load_data(TEST_IMAGE_DIR, TEST_NPY_DIR, LABELS)
test_image_files, test_hard_labels, _ = load_data_nonpy(TEST_IMAGE_DIR, MAX_PER_CLASS, 0, LABELS)
test_image_files = tf.constant(test_image_files, dtype=tf.string)


test_dataset = tf.data.Dataset.from_tensor_slices((test_image_files, test_hard_labels))
#test_dataset = test_dataset.shuffle(buffer_size=len(test_image_files)) # necesario poner esto antes que .batch()
test_dataset = test_dataset.map(parse_function_eval, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(bcolors.OKCYAN+f"Número de imágenes en test: {len(test_image_files)}"+bcolors.ENDC)
  
true_classes = np.array(test_hard_labels) 
predIdxs_prob = model.predict(test_dataset) #,steps=1 )
predIdxs = np.argmax(predIdxs_prob, axis=1)

#print(true_classes)
#print(predIdxs)

plot_confusion_matrix(true_classes, predIdxs, LABELS, FIGNAME=f'confusion_matrix{exp_sufix}.png')

from sklearn.metrics import classification_report
classificationReport = classification_report(true_classes, predIdxs, target_names=LABELS)
print(classificationReport)
plot_classification_report(classificationReport, cmap='viridis', FIGNAME=f'classification-report{exp_sufix}.png')

calculate_metrics(true_classes, predIdxs, predIdxs_prob)
plot_ROC(true_classes, predIdxs_prob, FIGNAME=f'ROC-{exp_sufix}.png')


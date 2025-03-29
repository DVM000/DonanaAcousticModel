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

from augmentdata import data_augmentation

#https://pyimagesearch.com/2021/06/28/data-augmentation-with-tf-data-and-tensorflow/


# ---------------------- PARAMETERS ---------------------- #
#image_path = "/repos/audio-birds/BirdNET-Analyzer/output/imgs/"  # Ruta de la carpeta con imágenes organizadas por categorías
#npy_path = "/repos/audio-birds/BirdNET-Analyzer/output/npy/"  # Ruta de la carpeta con archivos .npy de soft labels


TRAIN_IMAGE_DIR = "/tfm-external/birdnet-output_train/imgs/"  # Ruta de la carpeta con imágenes organizadas por categorías
TRAIN_NPY_DIR = "/tfm-external/birdnet-output_train//npy/"  # Ruta de la carpeta con archivos .npy de soft labels

VAL_IMAGE_DIR = "/tfm-external/birdnet-output_val/imgs/"  # Ruta de la carpeta con imágenes organizadas por categorías
VAL_NPY_DIR = "/tfm-external/birdnet-output_val/npy/"  # Ruta de la carpeta con archivos .npy de soft labels

TEST_IMAGE_DIR = "/tfm-external/birdnet-output_test/imgs/"  # Ruta de la carpeta con imágenes organizadas por categorías
TEST_NPY_DIR = "/tfm-external/birdnet-output_test/npy/"  # Ruta de la carpeta con archivos .npy de soft labels


TRAIN_IMAGE_DIR = "./tfm-external/less_classes/train/imgs/" #  Falco + Larus. 
VAL_IMAGE_DIR = "./tfm-external/less_classes/val/imgs/"
TEST_IMAGE_DIR = "./tfm-external/less_classes/test/imgs/"
TRAIN_NPY_DIR = "./tfm-external/less_classes/train/npy/"
VAL_NPY_DIR = "./tfm-external/less_classes/val/npy/"
TEST_NPY_DIR = "./tfm-external/less_classes/test/npy/"


MAX_PER_CLASS = 1000 # maximum data to take of each category
MIN_PER_CLASS = 50 # minimum data to take of each category

IMG_HEIGHT = 224 #64 # 128 # cuadrado 128x128?
IMG_WIDTH = 224 # 512
CHANNELS = 3
#NUM_CLASSES = 5 # obtained from Dataset
BATCH_SIZE = 32

rescaling = 1.0 / 255.0  # Normalización

INITIAL_LR = 5e-4 # 1st training
EPOCHS1 = 15

UNFREEZE = 50 # number of layers to unfreeze
FT_LR = 1e-4  # fine-tune
EPOCHS2 = 150

PATIENCE = 15

alpha = 0.9   #  0.5  # Peso de las hard labels en la mezcla (1-alpha = peso de soft labels)
temperature = 1.0  # Parámetro para suavizar soft labels


# ---------------------- CHECK GPU AVAILABLE ---------------------- #
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("\n WARNING: no GPU found. \n")
print(gpus)

# ---------------------- DATA LOADING ---------------------- #
def load_data(image_path, npy_path, category_list=[]):
    """
    Carga imágenes y soft labels desde carpetas organizadas por categorías.

    image_path: Ruta a la carpeta con imágenes organizadas en subdirectorios.
    npy_path: Ruta a la carpeta con archivos .npy organizados en subdirectorios.
    category_list: Lista deseada de categorias (subdirectorios a buscar). Si no se especifica, se toma de subdirectorios y se aplica MIN_PER_CLASS.

    Retorna listas de rutas de imágenes y sus etiquetas suaves.
    """
    image_files = []
    soft_labels = []
    hard_labels = []
    category_list_min = [] # lista de categorias que superan el minimo de datos
    
    # Recorrer categorías (subcarpetas)
    if len(category_list)<1:
        category_list = sorted(os.listdir(image_path)) # subcarpetas si no se especifica lista
        MIN_ = MIN_PER_CLASS 
    else:
        MIN_ = 0  # si se especifica lista, no usamos minimo
        
    ii = -1
    for i,category in enumerate(category_list):
        
        category_img_path = os.path.join(image_path, category)
        category_npy_path = npy_path #os.path.join(npy_path, category)

        if not os.path.isdir(category_img_path):
            continue  # Saltar archivos que no sean carpetas

        if len(os.listdir(category_img_path)) < MIN_: 
            print(f"Skipping category ({i}) {category}: only {len(os.listdir(category_img_path))} files")
            continue
        
        category_list_min.append(category)
        ii = ii + 1 # solo si la categoria ha sido añadida  
            
        # Recorrer las imágenes dentro de la categoría
        for img_file in os.listdir(category_img_path)[:MAX_PER_CLASS]:
            img_full_path = os.path.join(category_img_path, img_file)
            npy_full_path = os.path.join(category_npy_path, img_file.replace(".png", ".npy"))
            #print(img_full_path)
            #print(npy_full_path)
            
            if os.path.exists(npy_full_path):  # Asegurar que exista la predicción
                image_files.append(img_full_path)
                soft_labels.append(np.load(npy_full_path))  # Cargar soft label desde .npy
                hard_labels.append(ii)
                #print(img_full_path, i)
                     
        print(f"({i}) {category}: {min(len(os.listdir(category_img_path)[:MAX_PER_CLASS]), len(os.listdir(category_npy_path)[:MAX_PER_CLASS]))} files.")
                
    image_files = tf.constant(image_files, dtype=tf.string)
    #soft_labels = tf.convert_to_tensor(soft_labels, dtype=tf.float32)
    
    print(f"Found {len(image_files)} images belonging to {len(np.unique(hard_labels))} classes.")

    return image_files, np.array(soft_labels), hard_labels, category_list_min

# ---------------------- DATA PROCESSING ---------------------- #
def load_and_preprocess_image(image_path, label, augment=False):
    """
    Carga una imagen y la preprocesa para el entrenamiento.
    
    Parámetros:
    - image_path: Ruta de la imagen.
    - label: Etiqueta de la imagen.
    - augment: Booleano para aplicar o no data augmentation.
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
    
    # Asegurar que label es un tensor con la forma correcta
    label = tf.convert_to_tensor(label, dtype=tf.float32)
    #label.set_shape([num_classes])  # Establecer la forma explícitamente
    return image, label

def parse_function(image_path, label, augment=False):
    """
    Función de preprocesamiento usando `tf.numpy_function`.
    """
    # Necesitamos envolver las funciones de numpy en un tf.Tensor, para poder usarlas en el pipeline de tf.data.Dataset.
    image, label = tf.numpy_function(func=load_and_preprocess_image, 
                                     inp=[image_path, label, augment], 
                                     Tout=(tf.float32, tf.float32))

    image.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])  # Definir forma de imagen
    #label.set_shape([soft_labels.shape[1]])  
    #num_classes_total = soft_labels.shape[1] + hard_labels.shape[1] 
    #label.set_shape([num_classes_total])  # Definir número de clases *2

    return image, label
    
# Crear versiones separadas
def parse_function_train(image_path, label):
    return parse_function(image_path, label, augment=True)

def parse_function_eval(image_path, label):
    return parse_function(image_path, label, augment=False)
    
# ---------------------- DATASET ---------------------- #
# Cargar datos
print(f"Loading data...")
train_image_files, train_soft_labels, train_hard_labels, LABELS = load_data(TRAIN_IMAGE_DIR, TRAIN_NPY_DIR)
val_image_files, val_soft_labels, val_hard_labels, _ = load_data(VAL_IMAGE_DIR, VAL_NPY_DIR, LABELS)
print(train_image_files[:2])
    
# Las hard labels ahora mismo no estan codificadas en one-hot. Calculamos distribucion:
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
plt.savefig('DistribucionClases-distill.png')
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
    
with open(f'birdnet_idx.json', 'r') as fp:
    idx_dict = json.load(fp)
#print(idx_dict)

#idx = np.array(list(idx_dict.values()))  - 1
#print('\n ** WARNING: select correct soft labels ** \n\n ')
#idx = [2290-1, 2294-1, 2295-1, 2299-1, 2300-1, 2301-1]

# Select indices from idx_dict and LABELS:
idx = []
for l in LABELS:
    #print(l, idx_dict[l])
    idx.append(idx_dict[l]-1)
print(f"Selected indexes for our categories: {idx}")

# Guardar la lista final de especies 
with open('selected-species-model.txt', 'w') as f:
    for item in LABELS:
        f.write(item + "\n")
    
#print(np.argmax(train_soft_labels, axis=1))

train_soft_labels, train_hard_labels = selectNsoftlabels_and_gethardlabels(idx, train_soft_labels, train_hard_labels)
print("Soft labels examples:")
print(train_soft_labels[:2,:])
print("... that correspond to:")
print(np.argmax(train_soft_labels[:2,:], axis=1))
print("Hard labels examples:")
print(train_hard_labels[:2,:])

#sys.exit(0)

val_soft_labels, val_hard_labels = selectNsoftlabels_and_gethardlabels(idx, val_soft_labels, val_hard_labels)

# Convertir a tensores de TensorFlow
#train_image_files = tf.constant(train_image_files, dtype=tf.string)
#train_soft_labels = tf.convert_to_tensor(train_soft_labels, dtype=tf.float32)


#num_classes = soft_labels.shape[1]  # Se asume que las soft labels tienen el mismo número de clases
#train_hard_labels = np.eye(NUM_CLASSES)[train_hard_labels]  #  etiquetas one-hot
#print(NUM_CLASSES)

train_labels = np.concatenate([train_soft_labels, train_hard_labels], axis=1)
train_labels = tf.cast(train_labels, tf.float32)
#val_image_files = tf.constant(val_image_files, dtype=tf.string)
#val_soft_labels = tf.convert_to_tensor(val_soft_labels, dtype=tf.float32)
#val_hard_labels = np.eye(NUM_CLASSES)[val_hard_labels] 
val_labels = np.concatenate([val_soft_labels, val_hard_labels], axis=1)
val_labels = tf.cast(val_labels, tf.float32)
    
# ---------------------- DATASET ---------------------- #
#https://www.tensorflow.org/guide/data?hl=en
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_files, train_labels)) # 
train_dataset = train_dataset.shuffle(buffer_size=len(train_image_files)) # necesario poner esto antes que .batch()
train_dataset = train_dataset.map(parse_function_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
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
val_dataset = val_dataset.apply(tf.data.experimental.prefetch_to_device('/GPU:0')) 

# ---------------------- DEFINE MODEL ---------------------- #
base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Añadimos ultimas capas 'head'
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),  
    layers.Dense(NUM_CLASSES, activation='softmax')
])


print(model.input)
#sys.exit(0)

# ---------------------- DEFINIR PÉRDIDA DE DISTILLATION ---------------------- #
# ---------------------- MEZCLA DE HARD Y SOFT LABELS ---------------------- #
'''def distillation_loss(y_true, y_pred):
    """
    Pérdida combinada entre las etiquetas duras y suaves.
    """
    hard_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
    return hard_loss  # Si quieres incluir KL Divergence, agrégala aquí
'''
# https://keras.io/examples/vision/knowledge_distillation/

def distillation_loss(y_true, y_pred):
    """
    Pérdida combinada de crossentropy y KL Divergence.
    """
    num_classes = y_pred.shape[-1]  # Asegurar que num_classes es correcto
    hard_labels = y_true[:, -num_classes:]
    soft_labels = y_true[:, :num_classes] 
    #print(hard_labels); print(soft_labels); print(y_pred)
    
    # Convertir a float32 por estabilidad numérica
    soft_labels = tf.cast(soft_labels, tf.float32)
    hard_labels = tf.cast(hard_labels, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Asegurar que soft_labels y y_pred sean distribuciones de probabilidad
    soft_labels_softmax = tf.nn.softmax(soft_labels / temperature, axis=-1)
    y_pred_softmax = tf.nn.softmax(y_pred / temperature, axis=-1)  # Para KL Divergence

    # Cálculo de pérdidas
    kl_loss = tf.keras.losses.KLDivergence()(soft_labels_softmax, y_pred_softmax)
    hard_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(hard_labels, y_pred)
    #print("Hard Loss:", hard_loss.numpy(), "KL Loss:", kl_loss.numpy())

    # Mezcla de pérdidas
    total_loss = alpha * hard_loss + (1 - alpha) * kl_loss
    return total_loss

def custom_accuracy(y_true, y_pred):
    """
    Calcula la precisión usando solo las hard labels.
    """
    num_classes = tf.shape(y_pred)[-1]  # Número de clases dinámico

    # Extraer etiquetas duras (últimas columnas de y_true)
    hard_labels = y_true[:, -num_classes:]
    true_classes = tf.argmax(hard_labels, axis=-1)

    # Predicción del modelo
    pred_classes = tf.argmax(y_pred, axis=-1)

    # Comparar y calcular accuracy
    accuracy = tf.cast(tf.equal(true_classes, pred_classes), dtype=tf.float32)
    return tf.keras.backend.mean(accuracy)  # Promedio de aciertos

'''
print(train_labels[:10,:])  
print(train_labels[:10, -NUM_CLASSES:] )
print(distillation_loss(train_labels[:10,:], train_labels[:10, -NUM_CLASSES:]))
sys.exit(0)  '''  
  
# ---------------------- TRAIN MODEL ---------------------- #

# (1) Entrenar solo el 'head'
# ----------------------------------------------------------------------
base_model.trainable = False  # no entrenar inicialmente el modelo 
model.summary()

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=PATIENCE, restore_best_weights=True
)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
              loss=distillation_loss,
              #metrics=['categorical_accuracy'],) # cambiado
              metrics=[custom_accuracy]) #categorical_accuracy
  
history = model.fit(train_dataset, 
		  validation_data=val_dataset,
		  epochs=EPOCHS1,
		  #steps_per_epoch=200, # cambiado
		  #class_weight=class_weights,  # desalanceo de clases
                  callbacks=[early_stopping],
		  verbose=1)
		  

# (2) Entrenar mas capas
# ----------------------------------------------------------------------
for layer in model.layers:
    layer.trainable = True
    
for layer in (model.layers[0]).layers[:-UNFREEZE]:#[:100]:
    layer.trainable = False
        
model.summary()


lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=FT_LR, decay_steps=1000, decay_rate=0.96, staircase=True 
)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),#FT_LR),
              loss=distillation_loss,
              metrics=[custom_accuracy]) #sparse_categorical_accuracy
    
history_fine = model.fit(train_dataset, 
		  validation_data=val_dataset,
		  initial_epoch = EPOCHS1,
                  epochs= EPOCHS1+EPOCHS2, 
                  #steps_per_epoch=200, # cambiado
                  #class_weight=class_weights,
                  callbacks=[early_stopping],#, reduce_lr],
		  verbose=1)


def plot_history_1(acc,val_acc,loss,val_loss, namefig='fig1_distill.png'):
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

plot_history_1( list(history.history['custom_accuracy']) + list(history_fine.history['custom_accuracy']), 
	list(history.history['val_custom_accuracy']) + (history_fine.history['val_custom_accuracy']),
	list(history.history['loss']) + list(history_fine.history['loss']),
	list(history.history['val_loss']) + list(history_fine.history['val_loss']) )
	

# ---------------------- TEST MODEL ---------------------- #
model.save("mobilenet_spectrogram_distill.h5")

test_image_files, _, test_hard_labels, _ = load_data(TEST_IMAGE_DIR, TEST_NPY_DIR, LABELS)
test_image_files = tf.constant(test_image_files, dtype=tf.string)

test_dataset = tf.data.Dataset.from_tensor_slices((test_image_files, test_hard_labels))
#test_dataset = test_dataset.shuffle(buffer_size=len(test_image_files)) # necesario poner esto antes que .batch()
test_dataset = test_dataset.map(parse_function_eval, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Número de imágenes en test: {len(test_image_files)}")
  
true_classes = np.array(test_hard_labels) #test_generator.classes[test_generator.index_array].squeeze()  # esto debe ir antes que predict() 
predIdxs = model.predict(test_dataset) #,steps=1 )
predIdxs = np.argmax(predIdxs, axis=1)

#print(true_classes)
#print(predIdxs)


from sklearn.metrics import classification_report, confusion_matrix
print('Confusion Matrix')
cm = confusion_matrix(true_classes, predIdxs) #, labels=LABELS)
print(cm) 
print('Accuracy {:.2f}%'.format( 100*sum( (predIdxs.squeeze()==true_classes))/ true_classes.shape[0] ) ) 

from sklearn.metrics import ConfusionMatrixDisplay
cmP = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
fig, ax = plt.subplots(figsize=(60,60))
cmP.plot(ax=ax, colorbar=False)
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
plt.colorbar(cmP.im_,  cax=cax)
plt.show()
plt.savefig('confusion_matrix_distill.png')

classificationReport = classification_report(true_classes, predIdxs, target_names=LABELS)
print(classificationReport)

#https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
import itertools
import re

def plot_classification_report(classificationReport,
                               title='Classification report',
                               cmap='RdBu'):

    classificationReport = classificationReport.replace('\n\n', '\n')
    classificationReport = classificationReport.replace(' / ', '/')
    lines = classificationReport.split('\n')

    classes, plotMat, support, class_names = [], [], [], []

    for line in lines[1:-4]:  # Excluir la última parte con los promedios
        match = re.match(r"(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", line)
        if match:
            class_name, precision, recall, f1, sup = match.groups()
            classes.append(class_name.strip())
            class_names.append(class_name.strip())
            plotMat.append([float(precision), float(recall), float(f1)])
            support.append(int(sup))

    plotMat = np.array(plotMat)
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup)
                   for idx, sup in enumerate(support)]

    plt.figure(figsize=(10, 20))
    plt.imshow(plotMat, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(3), xticklabels, rotation=45)
    plt.yticks(np.arange(len(classes)), yticklabels)

    upper_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 8
    lower_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 2
    for i, j in itertools.product(range(plotMat.shape[0]), range(plotMat.shape[1])):
        plt.text(j, i, format(plotMat[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if (plotMat[i, j] > upper_thresh or plotMat[i, j] < lower_thresh) else "black")

    plt.ylabel('Classes')
    plt.xlabel('Metrics')
    plt.tight_layout()
    plt.show()
    plt.savefig('classification-report-distill.png')
 
plot_classification_report(classificationReport, cmap='viridis')
   


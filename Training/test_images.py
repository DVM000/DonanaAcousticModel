import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import sys
import json
import matplotlib.pyplot as plt
import tqdm
import random 

from  util import bcolors, check_GPU, plot_confusion_matrix, plot_classification_report, calculate_metrics, plot_ROC, test_check_data


# ---------------------- PARAMETERS ---------------------- #
MODEL_PATH = "../Models/mobilenet_spectrogram-all305-224.h5" 
MODEL_PATH = "../Models/mobilenet_spectrogram-all305-224-add-ft.h5"
#MODEL_PATH = "../Models/mobilenet_spectrogram-all305-128-add-ft.h5" 

TEST_IMAGE_DIR = "./tfm-external/less_classes/test/imgs/" # Ruta con subcarpetas de imagenes por cateogorias
TEST_NPY_DIR = "./tfm-external/less_classes/test/npy/"  # Ruta a archivos .npy de soft labels

TEST_IMAGE_DIR = "./AUDIOSTFM/test_imgs/"

MAX_PER_CLASS = 1000 # maximum data to take of each category
MIN_PER_CLASS = 50 # minimum data to take of each category

IMG_HEIGHT = 224 #128 
IMG_WIDTH = IMG_HEIGHT
CHANNELS = 3

BATCH_SIZE = 32

rescaling = 1.0 / 255.0  # Normalización

#TH_CONF = 0.5  # Umbral de confianza minima

# Solo testeamos en clases entrenadas por el modelo
SELECTED_SPECIES_FILE = "../Models/species-list-305.txt" # set to "" to take list of subdirectories
SELECTED_SPECIES_FILE = "../Models/species-list-337.txt" # set to "" to take list of subdirectories
#SELECTED_SPECIES_FILE = "" 

exp_sufix = '-prueba' # sufix for plot figures generated in this experiment


# ---------------------- LOAD TRAINED MODEL ---------------------- #
print(f"Loading model...")

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


model = load_model(MODEL_PATH, custom_objects={"distillation_loss": distillation_loss, "custom_accuracy": custom_accuracy})

model.summary()
print(model.input)

# ---------------------- DATA LOADING ---------------------- #
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
    

# ---------------------- SELECT LABELS ---------------------- # 
'''with open(f'birdnet_idx.json', 'r') as fp:
    idx_dict = json.load(fp)'''

try:
    with open(SELECTED_SPECIES_FILE, "r") as f:
         LABELS = [line.strip() for line in f]
except:
    LABELS = sorted(os.listdir(TEST_IMAGE_DIR))
    
print(LABELS, len(LABELS))

'''# Select indices from idx_dict and LABELS:
idx = []
for l in LABELS:
    #print(l, idx_dict[l])
    idx.append(idx_dict[l]-1)
print(f"Selected indexes for our categories: {idx}")'''


#LABELS = LABELS #+ sorted(os.listdir("AUDIOSTFM/fewshot_imgs/train"))

print(f"Target categories {LABELS}")
NUM_CLASSES = len(LABELS)

# ---------------------- TEST MODEL ---------------------- #
print(f"Loading data...")

test_image_files, test_hard_labels, _ = load_data_nonpy(TEST_IMAGE_DIR, MAX_PER_CLASS, 0, LABELS)
test_image_files = tf.constant(test_image_files, dtype=tf.string)

test_dataset = tf.data.Dataset.from_tensor_slices((test_image_files, test_hard_labels))
#test_dataset = test_dataset.shuffle(buffer_size=len(test_image_files)) # necesario poner esto antes que .batch()
test_dataset = test_dataset.map(parse_function_eval, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Número de imágenes en test: {len(test_image_files)}")
  
true_classes = np.array(test_hard_labels) 
predIdxs_prob = model.predict(test_dataset)#, steps=20); true_classes = true_classes[:20*32]
predIdxs = np.argmax(predIdxs_prob, axis=1)

#print(true_classes)
#print(predIdxs)

test_check_data(true_classes, predIdxs_prob, LABELS)

plot_confusion_matrix(true_classes, predIdxs, LABELS, FIGNAME=f'confusion_matrix-{exp_sufix}.png')

from sklearn.metrics import classification_report
classificationReport = classification_report(true_classes, predIdxs, target_names=LABELS)
print(classificationReport)
plot_classification_report(classificationReport, topN=25, cmap='viridis', FIGNAME=f'classification-report-{exp_sufix}.png')


calculate_metrics(true_classes, predIdxs, predIdxs_prob)
plot_ROC(true_classes, predIdxs_prob)

'''
# ---- From ImageDataGenerator
print('ImageDataGenerator')
from tensorflow.keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=rescaling, preprocessing_function=None)
test_generator = test_datagen.flow_from_directory(
    TEST_IMAGE_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)
    
true_classes = test_generator.classes[test_generator.index_array].squeeze()  # esto debe ir antes que predict() 
predIdxs = model.predict_generator(test_generator,steps=None) #,steps=1 )
predIdxs = np.argmax(predIdxs, axis=1)
print(true_classes)
print(predIdxs)

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

print(classification_report(true_classes, predIdxs, target_names=LABELS))'''


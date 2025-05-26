# https://github.com/hosodakazufumi/tfdone/blob/main/done_example_colab.ipynb

# NOTAS
# - Requiere una capa ‘Dense’ (fully-connected) al final de la red. 
#    * Si la arquitectura no la contiene, se pueden editar las capas finales de la misma para adaptarlo. 
#    * Si la arquitectura tiene una capa final ‘Softmax’, se puede reemplazar por una ‘Dense’ con activación
#        softmax: x = Dense(n_neurons, activation='softmax')(x), o bien simplemente eliminar la softmax, usar tf-DONE y luego añadirla de nuevo.
# - También permite re-ajustar los pesos de la última capa de la red (transfer learning) en lugar de añadir nuevas categorías/salidas. reconstruct=1
# - Para valores [0,1], la salida hay que escalarla dividiendo por la suma: y=y/y.sum() -- el modelo generado no tiene softmax --


import sys
import os
import numpy as np
import tqdm

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

import tensorflow as tf
from tfdone import done

from  util import bcolors

print('successfully imported')
print('In case of error do: export CUDA_VISIBLE_DEVICES=-1')

#  ---------------------- PARAMETERS ---------------------- #
MODEL_PATH = "mobilenet_spectrogram-all305-224.h5"
MODEL_PATH = "mobilenet_spectrogram-all305-128.h5"

TRAIN_IMAGE_DIR = "./tfm-external/less_classes/nuevas/" # Ruta con subcarpetas de imagenes por cateogorias
TEST_IMAGE_DIR = "./tfm-external/less_classes/test/imgs/" 

TRAIN_IMAGE_DIR = "./AUDIOSTFM/train_fshot/" 
TEST_IMAGE_DIR = "./AUDIOSTFM/test_imgs/"

MAX_PER_CLASS = 100 # maximum data to take of each category
MIN_PER_CLASS = 1 # minimum data to take of each category

IMG_HEIGHT = 128 #128 
IMG_WIDTH = IMG_HEIGHT 
CHANNELS = 3

BATCH_SIZE = 32

rescaling = 1.0 / 255.0  # Normalización


# ---------------------- LOAD TRAINED MODEL ---------------------- #
print(f"Loading model...")

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


model = load_model(MODEL_PATH, custom_objects={"distillation_loss": distillation_loss, "custom_accuracy": custom_accuracy})

model.summary()
print(model.input)

insize = model.input_shape[1:3]
IMG_SIZE = insize[0]
print('model input shape =', model.input_shape)
print('model output shape =', model.output_shape)



# ---------------------- DATA LOADING ---------------------- #
def load_data(image_path, category_list=[]):
    """
    Carga imágenes y soft labels desde carpetas organizadas por categorías.

    image_path: Ruta a la carpeta con imágenes organizadas en subdirectorios.
    category_list: Lista deseada de categorias (subdirectorios a buscar). Si no se especifica, se toma de subdirectorios y se aplica MIN_PER_CLASS.

    Retorna listas de rutas de imágenes y sus etiquetas.
    """
    image_files = []
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
            image_files.append(img_full_path)
            hard_labels.append(ii)
                
        print(f"({i}) {category}: {len(os.listdir(category_img_path)[:MAX_PER_CLASS])} files.")
               
    print(f"Found {len(image_files)} images belonging to {len(np.unique(hard_labels))} classes.")

    return image_files, hard_labels, category_list_min

# ---------------------- DATA PROCESSING ---------------------- #
def load_and_preprocess_list_of_images(image_files, augment=False, batch_size=32):
    """
    Carga una imagen y la preprocesa para el entrenamiento.
    
    Parámetros:
    - image_files: Rutas de la imagenes
    - augment: Booleano para aplicar o no data augmentation.
    
    Devuelve:
    - array de imagenes preprocesadas.
    """
    out = []
    for i in tqdm.tqdm(range(0, len(image_files), batch_size)):
        batch_paths = image_files[i:i+batch_size]
        batch_out = []   
        for image_path in batch_paths:
            
            image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))  # Cargar imagen
            image = img_to_array(image).astype(np.float32) * rescaling  # Convertir a array y normalizar
            if augment: # solo para training
                image = data_augmentation(image) #  Add data augmentation
                       
            image = np.expand_dims(image, axis=0)  # Añadir batch dimension
            batch_out.append(image)

        out.append(np.vstack(batch_out))

    return np.vstack(out)


print(f"Loading data...")
train_image_files, train_labels, LABELS = load_data(TRAIN_IMAGE_DIR)

print('Training_labels: {}'.format(train_labels))

train_processed = load_and_preprocess_list_of_images(train_image_files); print('[tf pre-processing for Train dataset]')
print('train_images_processed shape (num_images, height, width, channels) = {}'.format(train_processed.shape))

print('Adding training images, such as: ')
for i in np.random.randint(0,len(train_image_files),size=10):
    print(' {}\t(class {})'.format(train_image_files[i],train_labels[i]))

print('new images shape (num_images, height, width, channels) =', train_processed.shape)
print('add_y shape (num_images) =', np.array(train_labels).shape)
print('add_y labels: ', train_labels)


print(train_processed[0].mean()); print(train_processed[1].mean()); print(train_processed[1].max(), train_processed[1].min())
 
# ---------------------- SELECT LABELS ---------------------- # 
print(f"Target categories {LABELS}")
NUM_CLASSES = len(LABELS)

# Guardar la lista final de especies usadas para entrenar el modelo
SELECTED_SPECIES_FILE = f'selected-species-model.txt'
with open(SELECTED_SPECIES_FILE, "r") as f:
     LABELS_ORIG = [line.strip() for line in f]

ALL_LABELS = LABELS_ORIG + LABELS
selected_species_file2 = f'selected-species-model-add.txt'
with open(selected_species_file2, 'w') as f:
    for item in ALL_LABELS:
        f.write(item + "\n")
print(bcolors.OKCYAN+ f"Saved {selected_species_file2} with species from {SELECTED_SPECIES_FILE} + new ones." +bcolors.ENDC)   


# ---------------------- TRAIN MODEL ---------------------- #


########## Class addition by DONE ##########
REC=0
my_model_add = done.add_class(model, train_processed, np.array(train_labels), reconstruct=REC)
print('It\'s DONE! New classed added; model_add output shape = {}'.format(my_model_add.output_shape))
print('Using reconstruct={}'.format(REC))

my_model_add.summary()

# SAVE MODEL AS .h5: 
SAVE_PATH = MODEL_PATH.replace('.h5','')+'-add.h5'
#my_model_add.save( SAVE_PATH )
my_model_add.save( SAVE_PATH )
print(bcolors.OKCYAN+ '[Info] SAVED AS ' + SAVE_PATH + bcolors.ENDC)


sys.exit(0)

# Note that our original trained 'model' does not contain softmax layer. Add it here for testing:
#softmax_layer = tf.keras.layers.Softmax()
#softmax_prob = softmax_layer(my_model_add.output) 
#softmaxmodel = tf.keras.models.Model(inputs=my_model_add.input, outputs=softmax_prob)

#a = model.predict(train_processed[:1,:,:,:])
#print(np.max(a), np.sum(a))

#b = my_model_add.predict(train_processed[:1,:,:,:])
#print(np.max(b), np.sum(b))

#c = softmaxmodel.predict(train_processed[:1,:,:,:])
#print(np.max(c), np.sum(c))

# ---------------------- TEST MODEL ---------------------- #
print(f"Loading data...")
test_image_files, test_hard_labels, LABELS = load_data(TEST_IMAGE_DIR)

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
        
    image = tf.cast(image, tf.float32)  # Asegurar float32 después de la data augmentation'''
    
    ''''image = tf.io.read_file(image_path)
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

test_image_files = tf.constant(test_image_files, dtype=tf.string)

test_dataset = tf.data.Dataset.from_tensor_slices((test_image_files, test_hard_labels))
test_dataset = test_dataset.map(parse_function_eval, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Número de imágenes en test: {len(test_image_files)}")
  
true_classes = np.array(test_hard_labels) 
predIdxs = my_model_add.predict(test_dataset) 
predIdxs = np.argmax(predIdxs, axis=1)

true_classes = true_classes  
print(true_classes)
print(predIdxs)


from sklearn.metrics import classification_report, confusion_matrix
print('Confusion Matrix')
cm = confusion_matrix(true_classes, predIdxs) #, labels=LABELS)
print(cm) 
print('Accuracy {:.2f}%'.format( 100*sum( (predIdxs.squeeze()==true_classes))/ true_classes.shape[0] ) ) 

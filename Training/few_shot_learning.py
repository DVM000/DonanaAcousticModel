import sys
import os
import numpy as np
import tqdm

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model

from scipy.special import softmax

import matplotlib.pyplot as plt
import pickle

from  util import bcolors, check_GPU, plot_confusion_matrix, plot_classification_report, calculate_metrics, plot_ROC, test_check_data

#  ---------------------- PARAMETERS ---------------------- #
MODEL_PATH = "mobilenet-224-305.h5"
#MODEL_PATH = "mobilenet-128-305.h5"

TRAIN_IMAGE_DIR = "./tfm-external/less_classes/train/imgs/" # Ruta con subcarpetas de imagenes por cateogorias
TEST_IMAGE_DIR = "./tfm-external/less_classes/test/imgs/" 

TRAIN_IMAGE_DIR = "./AUDIOSTFM/train_imgs/" 
TEST_IMAGE_DIR = "./AUDIOSTFM/test_imgs/" 
#TEST_IMAGE_DIR = "./AUDIOSTFM/train_fshot/"

MAX_PER_CLASS = 100 # maximum data to take of each category
MIN_PER_CLASS = 1 # minimum data to take of each category

IMG_HEIGHT = 224 #128
IMG_WIDTH = IMG_HEIGHT
CHANNELS = 3

BATCH_SIZE = 32

rescaling = 1.0 / 255.0  # Normalización

# Embeddings extracting or loading
LOADpkl = False
pklfile = 'prototypes-224.pkl'
#pklfile = 'prototypes-128.pkl'

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
def load_data(image_path, MAX_PER_CLASS, MIN_PER_CLASS, category_list=[]):
    """
    Carga imagenes.

    image_path: Ruta a imagenes en subcarpetas.
    category_list: Lista deseada de categorias (subdirectorios a buscar). Si no se especifica, se toma de subdirectorios.

    Devuelve listas de rutas de imagenes y sus etiquetas.
    """
    image_files = []
    hard_labels = []
    category_list_min = [] # lista de categorias que superan el minimo de datos
    
    # Recorrer categorías (subcarpetas)
    if len(category_list)<1:
        category_list = sorted(os.listdir(image_path)) # subcarpetas si no se especifica lista
        
    ii = -1
    for i,category in enumerate(category_list):
        
        category_img_path = os.path.join(image_path, category)

        if not os.path.isdir(category_img_path):
            continue  # Saltar archivos que no sean carpetas

        if len(os.listdir(category_img_path)) < MIN_PER_CLASS: 
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
               
    print(bcolors.OKCYAN+ f"Found {len(image_files)} images belonging to {len(np.unique(hard_labels))} classes." +bcolors.ENDC)

    return image_files, hard_labels, category_list_min

# Para evitar memory issues:
def compute_prototypes_classwise(feature_model, image_path, MAX_PER_CLASS, MIN_PER_CLASS, category_list):
    """
    Carga imagenes desde subcarpetas, preprocesa imagenes y calcula prototipos por categoria.
    Eficiente en terminos de memoria.

    image_path: Ruta a imagenes en subcarpetas.
    category_list: Lista deseada de categorias (subdirectorios a buscar).

    Devuelve prototipos y sus etiquetas 
    """
    #all_labels = []
    prototypes = []
    #all_embeddings = []

    for i,category in enumerate(category_list):
        
        category_img_path = os.path.join(image_path, category)

        if not os.path.isdir(category_img_path):
            continue  # Saltar archivos que no sean carpetas

        image_files = []
        # Recorrer las imágenes dentro de la categoría
        for img_file in os.listdir(category_img_path)[:MAX_PER_CLASS]:
            img_full_path = os.path.join(category_img_path, img_file)
            image_files.append(img_full_path)
            #hard_labels.append(i)

        if len(image_files[-1]) == 0:
            print(f"No images found for class {category}")
            continue
            
        print(f"({i}) {category}: {len(os.listdir(category_img_path)[:MAX_PER_CLASS])} files.")

        # Pre-process and then free memory
        images = load_and_preprocess_list_of_images(image_files)
        embeddings = feature_model.predict(images, verbose=False)
        prototype = np.mean(embeddings, axis=0)

        prototypes.append(prototype)

        del images, image_files, embeddings  # Libera memoria inmediatamente
        #tf.keras.backend.clear_session()

    return (np.stack(prototypes), np.arange(len(prototypes)) )   #np.vstack(all_embeddings), 


# ---------------------- DATA PROCESSING ---------------------- #
def load_and_preprocess_list_of_images(image_files, augment=False, batch_size=32):
    """
    Carga imagenes y las preprocesa para el entrenamiento.
    
    Parametros:
    - image_files: ruta de las imagenes.
    - augment: aplicar o no data augmentation.
    """
    out = []
    for i in range(0, len(image_files), batch_size):
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


# --> MEMORY ISSUES
'''print(f"Loading data...")
train_image_files, train_labels, LABELS = load_data(TRAIN_IMAGE_DIR, MAX_PER_CLASS, 1)

train_processed = load_and_preprocess_list_of_images(train_image_files); 
print(bcolors.OKCYAN+ 'train_images_processed shape (num_images, height, width, channels) = {}'.format(train_processed.shape) +bcolors.ENDC)
     
# ---------------------- LABELS ---------------------- # 
#with open("species-list-305.txt", "r") as f:
#   LABELS = [line.strip() for line in f]
#print(LABELS)

print(f"Target categories {LABELS}")
NUM_CLASSES = len(LABELS)

# ---------------------- TRAIN MODEL ---------------------- #

# Embeddings extracting
feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
feature_model.summary()

# Función para extraer embeddings
def get_embeddings(img_batch):
    return feature_model.predict(img_batch)

# Calcular prototipos 
def compute_prototypes(support_images, support_labels):
    embeddings = get_embeddings(support_images)  # shape: (n_support, embedding_dim)
    classes = tf.unique(support_labels).y
    prototypes = []
    for c in classes:
        class_mask = tf.equal(support_labels, c)
        class_embeddings = tf.boolean_mask(embeddings, class_mask)
        prototype = tf.reduce_mean(class_embeddings, axis=0)
        prototypes.append(prototype)
    return embeddings, tf.stack(prototypes), classes
    
embeddings, prototypes, classes = compute_prototypes(train_processed, train_labels)
# <--  '''

# ---------------------- LOAD DATA & TRAIN MODEL ---------------------- #

# Model
feature_model = Model(inputs=model.input, outputs=model.layers[-2].output) #[-4]
feature_model.summary()

# Data loading
LABELS = sorted(os.listdir(TRAIN_IMAGE_DIR))
print(f"Target categories {LABELS}")
NUM_CLASSES = len(LABELS)


# Loading or saving prototype
if not LOADpkl:
    print(f"Loading images and Computing prototypes...")
    prototypes, classes = compute_prototypes_classwise(feature_model, TRAIN_IMAGE_DIR, MAX_PER_CLASS, 1, LABELS)

    with open(pklfile, 'wb') as f:
        pickle.dump({'prototypes': prototypes, 'classes': classes, 'labels': LABELS}, f)
else:     
    print(f"Loading pre-computed prototypes")
    with open(pklfile, 'rb') as f:
        data = pickle.load(f)
    prototypes = data['prototypes']
    classes = data['classes']
    LABELS = data['labels']
print(prototypes.shape)

# ---------------------- TEST MODEL ---------------------- #

# Función para extraer embeddings
def get_embeddings(img_batch):
    return feature_model.predict(img_batch)

# Clasificar imágenes por distancia a prototipos
def classify_queries(query_images, prototypes, classes):
    query_embeddings = get_embeddings(query_images)
    # Expand dims para broadcasting: (n_queries, 1, embedding_dim) - (1, n_classes, embedding_dim)
    dists = tf.norm(tf.expand_dims(query_embeddings, 1) - tf.expand_dims(prototypes, 0), axis=2)
    pred_indices = tf.argmin(dists, axis=1)
    predicted_labels = tf.gather(classes, pred_indices)
    return predicted_labels
    
print(f"\nLoading testing data...")
# --> MEMORY ISSUES
'''test_image_files, test_labels, LABELS = load_data(TEST_IMAGE_DIR, 20, 1)
test_processed = load_and_preprocess_list_of_images(test_image_files); 
print(bcolors.OKCYAN+ 'test_images_processed shape (num_images, height, width, channels) = {}'.format(test_processed.shape) +bcolors.ENDC)
query_images = test_processed
predicted_labels = classify_queries(query_images, prototypes, classes)

true_classes = np.array(test_labels) 
predIdxs = np.array(predicted_labels)
# <--'''


def classify_queries_streaming(feature_model, prototypes, classes, image_path, category_list, MAX_PER_CLASS=100, batch_size=32):
    """
    Carga imagenes desde subcarpetas, preprocesa imagenes, calcula embeddings y clasifica usando prototipos.
    Eficiente en terminos de memoria.

    image_path: Ruta a imagenes en subcarpetas.
    category_list: Lista deseada de categorias (subdirectorios a buscar).
    prototypes: prototipos por clase. Tamano (c,d)=(numero de clases, dimension de embeddings)
    classes: clases de los prototipos

    Devuelve:
      y_true: clase de las imagenes de test
      y_pred: prediccion basada en minima distancia a prototipos
      y_prob: vector suavizado de distancias: softmax(-distancias)
    """
    y_true = []
    y_pred = []
    y_prob = []

    for i,category in tqdm.tqdm(enumerate(category_list)):
        
        category_img_path = os.path.join(image_path, category)

        if not os.path.isdir(category_img_path):
            continue  # Saltar archivos que no sean carpetas

        image_files = []
        # Recorrer las imagenes dentro de la categoría
        for img_file in os.listdir(category_img_path)[:MAX_PER_CLASS]:
            img_full_path = os.path.join(category_img_path, img_file)
            image_files.append(img_full_path)
            #hard_labels.append(i)

        if len(image_files) == 0:
            print(f"No images found for class {category}")
            continue
            
        print(f"({i}) {category}: {len(os.listdir(category_img_path)[:MAX_PER_CLASS])} files.")
 
        # Process by batches
        for j in range(0, len(image_files), batch_size):
            batch_paths = image_files[j:j+batch_size]
            # Pre-process
            batch_imgs = load_and_preprocess_list_of_images(batch_paths, batch_size=len(batch_paths))
            # Embeddings
            batch_emb = feature_model.predict(batch_imgs, verbose=False)  # shape (batch_size, embedding_dim)

            # Distancias a prototipos: 
            dists = np.linalg.norm(batch_emb[:, None, :] - prototypes[None, :, :], axis=-1) # (b,1,d) - (1,c,d) = (b,c,d). norm along d-dimensions
            batch_pred = dists.argmin(axis=1) # index of minimum distance in matrix (b,c)

            # softmax sobre distancias negativas
            probs = softmax(-dists, axis=1)  # shape: (batch_size, num_classes)
        
            y_true.extend([i]*len(batch_pred))
            y_pred.extend(batch_pred)
            y_prob.extend(probs)

            # Liberar memoria
            del batch_imgs, batch_emb
            #tf.keras.backend.clear_session()

    return np.array(y_true), np.array(y_pred), np.array(y_prob)

y_true, y_pred, y_prob = classify_queries_streaming(feature_model, prototypes, classes, TEST_IMAGE_DIR, LABELS, 1000, batch_size=BATCH_SIZE)

true_classes = np.array(y_true) 
predIdxs = np.array(y_pred)
predIdxs_prob = np.array(y_prob)

#print(np.unique(true_classes))
#print(predIdxs_prob.shape)

#print(true_classes)
#print(predIdxs)

test_check_data(true_classes, predIdxs_prob, LABELS)

plot_confusion_matrix(true_classes, predIdxs, LABELS, FIGNAME='confusion_matrix-FS.png')

from sklearn.metrics import classification_report
classificationReport = classification_report(true_classes, predIdxs, target_names=LABELS)
print(classificationReport)
plot_classification_report(classificationReport, topN=20, cmap='viridis', FIGNAME='classification-report-FS.png')

calculate_metrics(true_classes, predIdxs, predIdxs_prob)
plot_ROC(true_classes, predIdxs_prob)

'''print(bcolors.WARNING+'With original model it was:'+bcolors.ENDC)
predIdxs = model.predict(query_images) #,steps=1 )
predIdxs = np.argmax(predIdxs, axis=1)
classificationReport = classification_report(true_classes, predIdxs, target_names=LABELS)
print(classificationReport)

calculate_metrics(true_classes, predIdxs, predIdxs_prob)'''


# ---------------------- PLOT EMBEDDINGS ---------------------- #
sys.exit(0)
# lo usé con el codigo que usaba el modelo de 11 clases, sin el codigo nuevo preparado para memory issues

embeddings = prototypes

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# embeddings -> (n_samples, d), prototypes -> (NUM_CLASSES, d)

# PCA 2 componentes
pca = PCA(n_components=2)
all_points = np.vstack([embeddings, prototypes])  # embeddings y prototipos
all_points_pca = pca.fit_transform(all_points)  # PCA

pca_embeddings = all_points_pca[:-NUM_CLASSES]  # embeddings
pca_prototypes = all_points_pca[-NUM_CLASSES:]  # prototipos


colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
plt.figure(figsize=(10, 7))

for i in range(NUM_CLASSES):
    class_points = pca_embeddings[np.array(train_labels) == i]  # Selección de los puntos por clase
    plt.scatter(class_points[:, 0], class_points[:, 1], color=colors[i], alpha=0.6)
    plt.scatter(pca_prototypes[i, 0], pca_prototypes[i, 1], color=colors[i], edgecolor='black',
                marker='o', s=200, label=f'Prototipo {i}')


#plt.title("Visualización de embeddings y prototipos (PCA)")
#plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('PCA.png')
print(bcolors.OKCYAN+'Saved as ' + f'PCA.png' +bcolors.ENDC)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# t-SNE 2 componentes
tsne = TSNE(n_components=2, random_state=42, perplexity=30)

all_points = np.vstack([embeddings, prototypes])
all_points_tsne = tsne.fit_transform(all_points) # t-SNE 

tsne_embeddings = all_points_tsne[:-NUM_CLASSES] # embeddings
tsne_prototypes = all_points_tsne[-NUM_CLASSES:] #prototipos


colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
plt.figure(figsize=(10, 7))

for i in range(NUM_CLASSES):
    class_points = tsne_embeddings[np.array(train_labels) == i] # Selección de los puntos por clase
    plt.scatter(class_points[:, 0], class_points[:, 1], color=colors[i], alpha=0.6)
    
    # centroides
    plt.scatter(tsne_prototypes[i, 0], tsne_prototypes[i, 1], color=colors[i], edgecolor='black',
                marker='o', s=200, label=f'Prototipo {i}')

#plt.title("Visualización de embeddings y prototipos (t-SNE)")
# plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('tSNE.png')
print(bcolors.OKCYAN+'Saved as ' + f'tSNE.png' +bcolors.ENDC)


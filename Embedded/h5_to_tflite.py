from tensorflow import keras, lite
from tensorflow.keras.models import load_model

# ---------------------- LOAD TRAINED MODEL ---------------------- #
def distillation_loss(y_true, y_pred):
    num_classes = y_pred.shape[-1]
    hard_labels = y_true[:, -num_classes:]
    soft_labels = y_true[:, :num_classes] 

    soft_labels = tf.cast(soft_labels, tf.float32)
    hard_labels = tf.cast(hard_labels, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    soft_labels_softmax = tf.nn.softmax(soft_labels / temperature, axis=-1)
    y_pred_softmax = tf.nn.softmax(y_pred / temperature, axis=-1)

    kl_loss = tf.keras.losses.KLDivergence()(soft_labels_softmax, y_pred_softmax)
    hard_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(hard_labels, y_pred)

    total_loss = alpha * hard_loss + (1 - alpha) * kl_loss
    return total_loss
  
def custom_accuracy(y_true, y_pred):
    num_classes = tf.shape(y_pred)[-1]
    hard_labels = y_true[:, -num_classes:]
    true_classes = tf.argmax(hard_labels, axis=-1)
    pred_classes = tf.argmax(y_pred, axis=-1)
    accuracy = tf.cast(tf.equal(true_classes, pred_classes), dtype=tf.float32)
    return tf.keras.backend.mean(accuracy)
    
      
def model_loading(MODEL_PATH):
    model = load_model(MODEL_PATH, custom_objects={"distillation_loss": distillation_loss, "custom_accuracy": custom_accuracy})
    #model.summary()
    return model


MODEL_PATH = "../Models/mobilenet-224-337wi-ft.h5" 
OUT_MODEL = MODEL_PATH.replace('.h5', '.tflite')
OUT_MODEL_QUANT = OUT_MODEL.replace('.tflite','-q.tflite')

model = model_loading(MODEL_PATH)
converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open(OUT_MODEL, 'wb') as f:
    f.write(tflite_model)
        
print('[INFO] Model saved as ' + OUT_MODEL )

# Cuantize the model --
converter = lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [lite.Optimize.DEFAULT]           # post-training dynamic range quantization
tflite_model = converter.convert()

with open(OUT_MODEL_QUANT, 'wb') as f:
    f.write(tflite_model)
        
print('[INFO] Model saved as ' + OUT_MODEL_QUANT )

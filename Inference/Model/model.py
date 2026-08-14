import tensorflow as tf
import os
import shutil
from classification_models.tfkeras import Classifiers

tf.keras.mixed_precision.set_global_policy("float32")
ResNet18, preprocess_input = Classifiers.get("resnet18")

inputs = tf.keras.layers.Input( shape=(224,224,3), dtype=tf.float32, name="input" )
x = tf.keras.layers.Lambda( lambda t: preprocess_input(tf.cast(t, tf.float32)), name="preprocess" )(inputs)

backbone = ResNet18( input_shape=(224,224,3), include_top=False, weights=None )

x = backbone(x)
x = tf.keras.layers.GlobalAveragePooling2D( name="gap" )(x)

embedding = tf.keras.layers.Lambda( lambda t: tf.cast(t, tf.float32), name="embedding" )(x)

probabilities = tf.keras.layers.Dense(  10,  activation="softmax",  dtype="float32",  name="probabilities" )(x)
predicted_class = tf.keras.layers.Lambda( lambda t: tf.argmax( t, axis=-1, output_type=tf.int32 ), name="predicted_class" )(probabilities)
model = tf.keras.Model(  inputs=inputs, outputs={  "probabilities": probabilities,  "embedding": embedding,  "predicted_class": predicted_class,  }  )

export_path="./model_repo/1"

if os.path.exists(export_path):
    shutil.rmtree(export_path)

model.export(export_path)

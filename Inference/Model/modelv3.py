import keras
import tensorflow as tf
from classification_models.tfkeras import Classifiers
from keras import layers

base_trained_model = keras.models.load_model("resnet18_68C_embedding_layer_v5_OK.keras")
embedding_output = base_trained_model.get_layer("embedding_norm").output
predicted_class_output = layers.Lambda(lambda x: tf.argmax(x, axis=1, output_type=tf.int32), name="predicted_class")(base_trained_model.output)

multi_output_model = keras.Model(
    inputs=base_trained_model.inputs, 
    outputs={"predicted_class": predicted_class_output, "embedding": embedding_output}
)

export_archive = keras.export.ExportArchive()
export_archive.track(multi_output_model) 

_, preprocess_input = Classifiers.get('resnet18')

@tf.function
def serve_fn(x):
    x_float = tf.cast(x, tf.float32) #cast to float32
    x_preprocessed = preprocess_input(x_float) #norm as in training
    return multi_output_model(x_preprocessed, training=False)

export_archive.add_endpoint(
    name="serving_default",
    fn=serve_fn,
    input_signature=[
        tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.uint8, name="input_image")
    ]
)

export_archive.write_out("resnet18_kserve_export")

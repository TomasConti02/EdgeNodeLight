import keras
import tensorflow as tf

print("Caricamento modello originale...")
base_trained_model = keras.models.load_model(
    "resnet18_68C_embedding_layer_v5_OK.keras"
)

# 1. Estraiamo l'output del layer 'embedding_norm' (512 dimensioni)
try:
    embedding_output = base_trained_model.get_layer("embedding_norm").output
except ValueError:
    # Fallback nel caso in cui il layer si chiami 'embedding_layer'
    embedding_output = base_trained_model.get_layer("embedding_layer").output

# 2. Creiamo un nuovo modello Keras con 2 output (probabilità + embedding)
multi_output_model = keras.Model(
    inputs=base_trained_model.inputs,
    outputs={
        "probabilities": base_trained_model.output,  # Shape: (None, 68)
        "embedding": embedding_output               # Shape: (None, 512)
    }
)

# 3. Inizializziamo ExportArchive per il modello multi-output
export_archive = keras.export.ExportArchive()
export_archive.track(multi_output_model)


# 4. Definizione della funzione di serving
@tf.function
def serve_fn(input_image):
    x = tf.cast(input_image, tf.float32)
    # Eseguiamo l'inferenza con training=False per usare i moving mean/var della BatchNorm
    preds = multi_output_model(x, training=False)
    
    return {
        "probabilities": preds["probabilities"],
        "embedding": preds["embedding"]
    }


# 5. Aggiunta dell'endpoint con firma di input corretta
export_archive.add_endpoint(
    name="serving_default",
    fn=serve_fn,
    input_signature=[
        tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32, name="input_image")
    ],
)

# 6. Scrittura dell'artifact per TF Serving / KServe
export_archive.write_out("resnet18_kserve_export")
print("Modello esportato con successo con doppio output: 'probabilities' (68) e 'embedding' (512)!")

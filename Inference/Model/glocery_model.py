import os
import shutil
import tensorflow as tf
from tensorflow import keras

model_path = "resnet18_68C_embedding_layer_v5_OK.keras"
print(f"Loading model from: {model_path}...")
loaded_model = keras.models.load_model(model_path)
print("Model loaded successfully!\n")

embedding_output = loaded_model.get_layer("embedding_norm").output
preds_output = loaded_model.get_layer("dense_1").output

export_model = keras.Model(
    inputs=loaded_model.input,
    outputs={
        "embedding": embedding_output,
        "predicted_class": preds_output
    },
    name="resnet_kserve_export"
)

temp_keras_path = "temp_export_model.keras"
export_model.save(temp_keras_path)

clean_model = keras.models.load_model(temp_keras_path)

export_path = "./model_repo/1"
if os.path.exists(export_path):
    shutil.rmtree(export_path)

clean_model.export(export_path)
print(f"Model successfully saved and exported to: {export_path}!\n")

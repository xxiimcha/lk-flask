import tensorflow as tf

# Load the trained .h5 model
model = tf.keras.models.load_model("plant_growth_model.h5")

# Convert to TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: enable optimizations (comment out if accuracy drops)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save to file
with open("plant_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion successful: plant_model.tflite")

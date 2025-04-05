from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="plant_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open('label_names.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = Image.open(file.stream).resize((150, 150)).convert('RGB')
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

    # Set input and run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_index = int(np.argmax(output))
    predicted_label = labels[predicted_index]
    confidence = float(output[predicted_index])

    return jsonify({'prediction': predicted_label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)

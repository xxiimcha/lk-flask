from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model('plant_growth_model.h5')

with open('label_names.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = Image.open(file.stream).resize((150, 150)).convert('RGB')
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]

    return jsonify({'prediction': predicted_label, 'confidence': float(prediction[predicted_index])})

if __name__ == '__main__':
    app.run(debug=True)

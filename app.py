from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = tf.keras.models.load_model('plant_growth_model.h5')

# Load label names from saved file
with open('label_names.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    image_url = ''
    
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess image
            img = Image.open(filepath).resize((150, 150)).convert('RGB')
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

            # Predict
            pred = model.predict(img_array)
            result = labels[np.argmax(pred)]
            image_url = filepath

    return render_template('index.html', result=result, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)

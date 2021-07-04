from flask import Flask, render_template, request

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import ImageFile

from tensorflow import keras

app = Flask(__name__)

model = keras.models.load_model('saved_model/my_model')

#model.make_predict_function()
class_names = ['Buku', 'Globe', 'Jam Dinding', 'Piring', 'Spidol', 'Tas']
OUTPUT_DIR = 'static/uploaded'
@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def worksheet():
    return render_template('worksheet.html')

@app.route('/mathdetect', methods=['POST', 'GET'])
def mathdetect():
    return render_template('predict.html')
    
@app.route('/', methods=['POST', 'GET'])
def predict():
    imagefile = request.files['file']
    image_path = os.path.join(OUTPUT_DIR, imagefile.filename)
    #image_path = imagefile.filename
    imagefile.save(image_path)

    image = keras.preprocessing.image.load_img(image_path, target_size=(200,200))
    image = keras.preprocessing.image.img_to_array(image)
 
    img_array = np.array([image])
    prediksi = model.predict(img_array)
    score = tf.nn.softmax(prediksi[0])
    if (class_names[np.argmax(score)]=='Buku'):
        return render_template('buku.html', prediction=class_names[np.argmax(score)], gambar=image_path)
    elif (class_names[np.argmax(score)]=='Spidol'):
        return render_template('spidol.html', prediction=class_names[np.argmax(score)], gambar=image_path)
    elif (class_names[np.argmax(score)]=='Piring'):
        return render_template('piring.html', prediction=class_names[np.argmax(score)], gambar=image_path)
    elif (class_names[np.argmax(score)]=='Jam Dinding'):
        return render_template('jamdinding.html', prediction=class_names[np.argmax(score)], gambar=image_path)
    elif (class_names[np.argmax(score)]=='Globe'):
        return render_template('globe.html', prediction=class_names[np.argmax(score)], gambar=image_path)
    elif (class_names[np.argmax(score)]=='Tas'):
        return render_template('tas.html', prediction=class_names[np.argmax(score)], gambar=image_path)
    else :
        return render_template('predict.html', gambar=logo)


if __name__ == '__main__':
    app.run(port=666, debug=True)
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
import keras
import os



app = Flask(__name__)
app_root = os.path.dirname(os.path.abspath(__file__))

@app.route('/', methods=['GET', 'POST'])
def main():
    target = os.path.join(app_root, 'static/img/')
    if not os.path.isdir(target):
        os.makedirs(target)
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        result = ""
        file = request.files['file']
        file_name = file.filename
        dest = '/'.join([target, file_name])
        file.save(dest)
        model = keras.models.load_model('model-development/model_2.h5')

        
        img = image.load_img(dest, target_size=(200, 200))
        x = image.img_to_array(img)
        plt.imshow(x/255.)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        print(classes[0])
        if classes[0]<0.5:
            result = "Your soil is in good condition!"
        else:
            result = "Your soil is not fertile"
        return render_template("index.html" , result=result)
    else:
        return "Unsupported Request Method"

if __name__ == '__main__':
    app.run(debug=True)
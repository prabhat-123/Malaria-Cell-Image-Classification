import os
import numpy as np
from tensorflow.keras.models import model_from_json
import json
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.python.keras.backend import set_session

tf.keras.backend.clear_session()
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


from flask import url_for,request,Flask,redirect,render_template
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename


# Define a Flask App
app = Flask(__name__)

# Define the type of cell you are going to predict
cell_type = ['Parasitized','Uninfected']

# Load Model
json_file = open('malariacell_classification_model.json','r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('malaria_cell_classification_model_18-0.956981.h5')
print("Model loaded from disk")

model.summary()
model._make_predict_function()


def model_predict(model,image_path):
    img = image.load_img(image_path,target_size=(224,224))
    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    # Rescaling the image
    x = x/255.0
    preds = model.predict(x)
    return preds

@app.route('/',methods=['GET'])
def index():
    return render_template('./index.html')

@app.route('/',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f = request.files['file']
        # save image to ./uploads folder
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path,'uploads',secure_filename(f.filename))
        f.save(file_path)

        # Make a prediction
        preds = model_predict(model,file_path)
        if preds > 0.5:
            pred_class = 1
        else:
            pred_class = 0
        result = cell_type[pred_class]
        return render_template('./predict.html',result=result)
    else:
        return render_template('./index.html')

if __name__ == '__main__':
    app.run(debug=True)





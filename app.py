import tensorflow
from flask import Flask, render_template, request
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import pickle5 as pickle

app = Flask(__name__)
model = load_model('model.h5')
# loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def predict_label(img_path):
	input_img = cv2.imread(img_path)
	input_img_resized = cv2.resize(input_img, (224,224))
	input_img_scaled = input_img_resized/255
	image_reshaped = np.reshape(input_img_scaled, [1,224,224,3])
	input_prediction = model.predict(image_reshaped)
	input_pred_label = np.argmax(input_prediction)

	if input_pred_label == 0:
		print('It\'s Cat')
	else:
		print('It\'s Dog')

# routes
@app.route('/', methods=['GET', 'POST'])
def main():
	return render_template('index.html')

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "/static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = input_pred_label, img_path = img_path)

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)

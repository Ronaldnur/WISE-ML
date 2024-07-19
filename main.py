import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from google.cloud import storage
from io import BytesIO


app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = set(['png', 'jpg', 'jpeg'])
app.config["UPLOAD_FOLDER"] = "static/uploads/"  # Direktori statik untuk menyimpan URL gambar

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'wise-credentials.json'
storage_client = storage.Client()

model = load_model("model.h5", compile=False)

with open("labels.txt", "r") as file:
    labels = file.read().splitlines()

class_names = ["Cardboard", "Glass", "Metal", "Paper", "Plastic"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        if 'image' not in request.files:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "No image part"
                },
                "data": None
            }), 400

        image = request.files['image']

        if image and allowed_file(image.filename):
            image_bytes = image.read()  # Baca konten gambar sebagai bytes
            destination_blob_name = f'static/uploads/{secure_filename(image.filename)}'

            # Mengunggah bytes ke Google Cloud Storage
            image_bucket = storage_client.bucket('wise-bucket')  # Ganti dengan nama bucket Anda
            blob = image_bucket.blob(destination_blob_name)
            blob.upload_from_string(image_bytes, content_type=image.content_type)

            image_url = blob.public_url

            # Menyimpan URL relatif ke direktori statik
            #relative_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)

            # Persiapan gambar untuk prediksi (jika diperlukan)
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.asarray(img)
            img_array = np.expand_dims(img_array, axis=0)
            normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = float(prediction[0][index])

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success predicting"
                },
                "data": {
                    "sampah_types_prediction": class_name,
                    "confidence": confidence_score,
                    "image_url": image_url,
                    #"relative_path": relative_path  # Mengirimkan path relatif untuk digunakan di frontend
                }
            }), 200

        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Invalid image format"
                },
                "data": None
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None
        }), 405

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
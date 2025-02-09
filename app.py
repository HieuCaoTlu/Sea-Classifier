import os
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import onnxruntime as ort
import gdown

app = Flask(__name__)
sea_creatures = ['Trai', 'San hô', 'Cua', 'Cá heo', 'Lươn',
 'Cá', 'Sứa', 'Tôm hùm', 'Sên biển', 'Bạch tuộc',
 'Rái cá', 'Chim cánh cừ', 'Cá nóc', 'Cá đuối', 'Cầu gai',
 'Cá ngựa', 'Cá mập', 'Tôm', 'Mực', 'Sao biển',
 'Rùa', 'Cá voi']
model_path = "model.onnx"
file_id = '13mSlO2eCOuU7DVA5ctC2JQ_J0ogbdfss'
destination = 'model.onnx'

def load_model():
    if not os.path.isfile(destination):
        gdown.download(f'https://drive.google.com/uc?id={file_id}', destination, quiet=True)
    return ort.InferenceSession(destination)

def preprocess_image(image, size=(260, 260)):
    image = image.resize(size)
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))  # Chuyển kênh ảnh về (C, H, W)
    return np.expand_dims(image_array, axis=0)  # Thêm batch dimension

def prediction(img_path):
    model = load_model()  # Load model here
    image = Image.open(img_path)
    input_tensor = preprocess_image(image)
    outputs = model.run(None, {model.get_inputs()[0].name: input_tensor})
    predicted_class = np.argmax(outputs[0])
    return sea_creatures[predicted_class]

def get_top_3_classes(img_path):
    model = load_model()  # Load model here
    image = Image.open(img_path)
    input_tensor = preprocess_image(image)
    outputs = model.run(None, {model.get_inputs()[0].name: input_tensor})
    probabilities = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]))
    top_classes = np.argsort(probabilities[0])[-3:][::-1]
    top_probs = probabilities[0][top_classes] * 100
    return [(sea_creatures[idx], int(prob)) for idx, prob in zip(top_classes, top_probs)]

@app.route('/', methods=['GET'])
def index():
    if os.path.isfile('object.jpg'):
        os.remove('object.jpg')
        os.remove('./static/object.jpg')
    return render_template('index2.html', appName="Sea Animals Classifier", image='./static/placeholder.jpg', decoration=True, result='Đang đợi nhập', crab=False)

@app.route('/', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify(success=False, message='No file part')
    
    file = request.files['image']
    if file.filename == '':
        return jsonify(success=False, message='No selected file')
    
    static_file_path = './static/object.jpg'
    img = Image.open(file.stream)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img.save(static_file_path, "JPEG")

    result = prediction(static_file_path)
    crab = result == "Cua"
    top_3_classes = get_top_3_classes(static_file_path)
    
    return render_template('index2.html', appName="Sea Animals Classifier", image=static_file_path, decoration=False, result=result, crab=crab, top_3=top_3_classes)

if __name__ == '__main__':
    app.run(debug=True)

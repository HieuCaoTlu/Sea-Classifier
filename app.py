import os
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import torch
import gdown

app = Flask(__name__)
sea_creatures = ["San hô", "Cua", "Cá heo", "Cá", "Sứa", "Tôm hùm", "Sên biển", "Bạch tuộc", "Rái cá", 
                 "Cá nóc", "Cá đuối", "Nhím biển", "Cá mập", "Sao biển", "Rùa", "Cá voi"]
model = None

def preprocess_image(image, size=(224, 224)):
    image = image.resize(size)
    image_array = np.array(image).astype(np.float32)
    image_array /= 255.0
    image_tensor = torch.tensor(image_array).permute(2, 0, 1)
    return image_tensor.unsqueeze(0)

def loader():
    global model
    file_id = '1HGbW5KMdge6s-Tc8LdKOxL-GRrRmNN1m'
    destination = 'model_scripted_2.pt'
    if not os.path.isfile(destination):
        gdown.download(f'https://drive.google.com/uc?id={file_id}', destination, quiet=False)
        if os.path.isfile(destination):
            model = torch.jit.load('model_scripted_2.pt')
        else:
            print('Lỗi không load được mô hình')
    else:
        model = torch.jit.load('model_scripted_2.pt')

def prediction(img_path):
    global model, sea_creatures
    animals = sea_creatures
    loader()
    image = Image.open(img_path)
    input_tensor = preprocess_image(image)
    model.to(torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    predicted_class = animals[predicted.item()]
    return predicted_class

def get_top_3_classes(img_path):
    global model, sea_creatures
    animals = sea_creatures
    loader()
    image = Image.open(img_path)
    input_tensor = preprocess_image(image)
    model.to(torch.device('cpu'))
    model.eval()
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probabilities, 3)

    return [(animals[idx.item()], prob.item()) for idx, prob in zip(top_classes[0], top_probs[0])]

@app.route('/', methods=['GET'])
def index():
    if os.path.isfile('object.jpg'):
        os.remove('object.jpg')
        os.remove('./static/object.jpg')
    return render_template('index.html', appName="Sea Animals Classifier", image='./static/placeholder.jpg', decoration=True, result = 'Đang đợi nhập', crab=False)


@app.route('/', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify(success=False, message='No file part')
    
    file = request.files['image']
    if file.filename == '':
        return jsonify(success=False, message='No selected file')
    
    static_file_path = './static/object.jpg'
    img = Image.open(file.stream)
    img.save(static_file_path, "JPEG")

    result = prediction(static_file_path)
    if result == "Cua": crab=True
    else: crab = False
    top_3_classes = get_top_3_classes(static_file_path)
    return render_template('index.html', appName="Sea Animals Classifier", image=static_file_path, decoration=False, result=result, crab=crab, top_3=top_3_classes)

if __name__ == '__main__':
    app.run(debug=True)
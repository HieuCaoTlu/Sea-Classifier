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
        gdown.download(f'https://drive.google.com/uc?id={file_id}', destination, quiet=True)
        model = torch.jit.load('model_scripted_2.pt')
    else:
        pass

def prediction(img_path, model, animals):
    image = Image.open(img_path)
    input_tensor = preprocess_image(image)
    model.to(torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    predicted_class = animals[predicted.item()]
    return predicted_class

@app.route('/', methods=['GET'])
def index():
    loader()
    if os.path.isfile('object.jpg'):
        os.remove('object.jpg')
        os.remove('./static/object.jpg')
    return render_template('index.html', appName="Sea Animals Classifier", image='./static/placeholder.jpg', decoration=True, result = 'Đang đợi nhập')


@app.route('/', methods=['POST'])
def upload_image():
    global model, sea_creatures
    if 'image' not in request.files:
        return jsonify(success=False, message='No file part')
    
    file = request.files['image']
    if file.filename == '':
        return jsonify(success=False, message='No selected file')
    
    # Đường dẫn đến file mới trong thư mục static
    static_file_path = './static/object.jpg'
    
    # Lưu ảnh vào thư mục static mà không xóa ảnh cũ
    img = Image.open(file.stream)
    img.save(static_file_path, "JPEG")

    # Thực hiện dự đoán
    loader()
    result = prediction(static_file_path, model, sea_creatures)
    
    # Trả về trang index với ảnh đã tải lên và kết quả dự đoán
    return render_template('index.html', appName="Sea Animals Classifier", image=static_file_path, decoration=False, result=result)

if __name__ == '__main__':
    app.run(debug=True)
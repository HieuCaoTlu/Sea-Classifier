# ⛵️ Sea Animal Classifier 🌊

## 🐠 Giới thiệu
Dự án này sử dụng mô hình **EfficientNet B2** để phân loại **22 loài sinh vật biển**. Bộ dữ liệu được lấy từ Kaggle: [Sea Animals Image Dataset](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste). Mô hình đạt **85% Accuracy** và **75% Precision**.

## 🌍 Công nghệ sử dụng
- **EfficientNet B2**: Mô hình CNN để phân loại ảnh.
- **Flask**: API backend để xử lý yêu cầu.
- **Render**: Dịch vụ cloud để deploy ứng dụng.

## 🛠️ Cài đặt & Chạy thử
### 1. Clone repository
```bash
git clone https://github.com/your-username/sea-animal-classifier.git
cd sea-animal-classifier
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Chạy ứng dụng Flask
```bash
python app.py
```
Ứng dụng sẽ chạy trên `http://127.0.0.1:5000/`

## 📡 Demo & Deploy
Ứng dụng đã được deploy trên Render. Bạn có thể dùng thử tại đây:
[Sea Classifier - Render](https://sea-classifier.onrender.com/)

## 🌟 Kết quả mô hình
- **Accuracy**: 85%
- **Precision**: 75%

---
Chúc bạn có một trải nghiệm vui vẻ với dự án! 🌟

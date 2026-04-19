# 🐾 Animal Classification System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Dự án này ứng dụng Deep Learning để tự động nhận diện và phân loại các loài động vật từ hình ảnh thông qua mạng nơ-ron tích chập (CNN).

---

## 📑 Mục lục
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Dataset](#-dataset)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cài đặt](#-cài-đặt)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
  - [Huấn luyện (Training)](#huấn-luyện-training)
  - [Kiểm thử (Inference)](#kiểm-thử-inference)
- [Model đã huấn luyện](#-model-đã-huấn-luyện)

---

## 💻 Công nghệ sử dụng

- **Ngôn ngữ:** Python
- **Framework:** PyTorch
- **Xử lý dữ liệu:** NumPy, OpenCV
- **Trực quan hóa:** TensorBoard

---

## 📸 Dataset

Mô hình được huấn luyện trên bộ dữ liệu **Animals-10** để nhận diện 10 loại động vật sau:

| 🦋 Butterfly | 🐱 Cat | 🐔 Chicken | 🐄 Cow | 🐶 Dog |
| :---: | :---: | :---: | :---: | :---: |
| 🐘 **Elephant** | 🐴 **Horse** | 🐑 **Sheep** | 🕷️ **Spider** | 🐿️ **Squirrel** |

📥 **[Tải Dataset tại đây](https://drive.google.com/file/d/1jmWsnUgL_XB1kkPwnGWAWu_I2E-qKiF5/view?usp=sharing)**

---

## 📂 Cấu trúc dự án

```text
📦 Classification_Animals
 ┣ 📂 Colab Notebooks      # Các file notebook thử nghiệm trên môi trường Colab
 ┣ 📂 tensorboard          # Thư mục lưu nhật ký huấn luyện (logs) để xem trên TensorBoard
 ┣ 📜 dataset_animal.py    # Định nghĩa cấu trúc Custom Dataset và DataLoader
 ┣ 📜 models.py            # Chứa kiến trúc mạng nơ-ron CNN
 ┣ 📜 train_cnn.py         # Script chính để huấn luyện mô hình
 ┣ 📜 test_cnn.py          # Script để đánh giá và kiểm thử dự đoán
 ┗ 📜 README.md            # Tài liệu hướng dẫn dự án
```

---

## ⚙️ Cài đặt

1. **Clone repository:**
   ```bash
   git clone https://github.com/EngineeRing-hub-lab/Classification_Animals.git
   cd Classification_Animals
   ```

2. **Cài đặt thư viện:**
   ```bash
   pip install numpy pandas opencv-python tensorboard matplotlib torch torchvision
   ```

---

## 🚀 Hướng dẫn sử dụng

### Huấn luyện (Training)
Để bắt đầu huấn luyện mô hình từ đầu, hãy đảm bảo bạn đã giải nén dataset vào thư mục dự án theo đúng đường dẫn và chạy lệnh sau:

```bash
python train_cnn.py
```
*💡 Lưu ý: Bạn có thể theo dõi biểu đồ Loss và Accuracy theo thời gian thực bằng cách mở một terminal mới và chạy lệnh `tensorboard --logdir=tensorboard`.*

### Kiểm thử (Inference)
Để sử dụng mô hình dự đoán trên các hình ảnh mới:

```bash
python test_cnn.py
```

---

## 🏆 Model đã huấn luyện (Pre-trained Model)

Nếu bạn muốn trải nghiệm hoặc kiểm thử ngay kết quả tốt nhất mà không cần mất thời gian chạy lại quá trình training:

🚀 **[Tải file trọng số `best.pt` tại đây](https://drive.google.com/file/d/1qeTH28upnl9yhPWNCOSSOeYWHjNfbylT/view?usp=sharing)**

**Cách tích hợp:**
1. Đặt file `best.pt` vừa tải về vào thư mục gốc của dự án.
2. Trong file mã nguồn kiểm thử (`test_cnn.py`), bạn có thể load trọng số này vào model như sau:

```python
import torch
from models import YourModelClass # Hãy thay thế bằng tên class model thực tế của bạn

# Khởi tạo model và tải trọng số đã huấn luyện
model = YourModelClass()
model.load_state_dict(torch.load('best.pt', map_location=torch.device('cpu')))
model.eval()

# Tiến hành dự đoán...
```
```

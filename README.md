
Khởi tạo môi trường
```bash
python -m venv venv
```



Cài đặt môi trường ứng dụng
Window
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

Linux:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

Tạo bản đồ, nhiệm vụ:
Window
```bash
venv\Scripts\activate
python -m data.generator
```

Linux:
Window
```bash
source venv/bin/activate
python -m data.generator
```

Train
```bash
venv\Scripts\activate
python -m train.trainer
```

Linux:
```bash
source venv/bin/activate
python -m train.trainer
```
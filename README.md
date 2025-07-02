# VisionAid-AppAsistenPenglihatanTunanetra

Aplikasi dari computer vision yang dikembangkan untuk membantu tunanetra



VisionAid adalah sistem asisten penglihatan berbasis AI yang membantu penyandang tunanetra dalam navigasi sehari-hari dengan mendeteksi objek sekitar dan memberikan umpan balik suara secara real-time.



##### 🌟 Fitur Utama

* Deteksi Objek Real-time

Mengenali 80+ objek sehari-hari menggunakan model YOLO/MobileNet



* Estimasi Jarak

Menghitung jarak objek dengan teknik geometri monocular



* Umpan Balik Suara

Konversi hasil deteksi ke panduan audio dalam bahasa Indonesia



* Multi-sumber Kamera

Dukungan kamera USB, IP kamera, dan smartphone



* Antarmuka Responsif

Desain web yang mudah diakses dengan feedback visual jelas

##### 

##### 🛠 Teknologi

1. Backend

* Python 3.8+
* Flask (Web Framework)
* OpenCV (Computer Vision)
* PyTorch (Deep Learning)
* Ultralytics YOLOv8 (Object Detection)
* pyttsx3 (Text-to-Speech)



**2. Frontend**

* Bootstrap 5 (UI Components)
* Socket.IO (Real-time Communication)
* Vanilla JavaScript



##### Langkah-langkah

* ###### Clone repository

git clone https://github.com/username/VisionAid.git

cd VisionAid

* ###### Buat environment virtual (disarankan)

python -m venv venv

source venv/bin/activate  # Linux/Mac

venv\\Scripts\\activate    # Windows

* ###### Instal dependensi

pip install -r requirements.txt

* ###### Unduh model AI

python download\_models.py

* ###### Jalankan aplikasi

python app.py





##### Model  yang Didukung

* YOLOv8 Nano - Akurasi tinggi (65.2% mAP)
* MobileNet-SSD - Kecepatan optimal (78 FPS)
* YOLOv5 Nano - Kompatibilitas luas

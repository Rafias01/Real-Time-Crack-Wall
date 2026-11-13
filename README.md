# Wall Crack Detection Using YOLO11

Project Computer Vision untuk mendeteksi lokasi retakan dinding dengan bounding box menggunakan YOLO11n. Training di Google Colab GPU, inferensi melalui Streamlit di laptop RAM 8 GB. Model tidak melakukan segmentasi atau klasifikasi tingkat bahaya.

## Dataset
- ADAS, *wall crack*, Roboflow Universe, versi 1: https://universe.roboflow.com/adas-fvpqc/wall-crack-p0jqt/dataset/1
- Lisensi dataset: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
- 5.169 gambar pada versi sumber, satu kelas; tanpa augmentasi tambahan pada versi tersebut. Preprocessing sumber: auto-orient, resize stretch 512×512.
- Seluruh gambar sumber berada pada train. Notebook mengaudit label dan menghapus duplikat persis, lalu membagi 80/10/10 dengan seed 42. Jumlah unik belum diketahui. Near-duplicate dan foto dari dinding yang sama memerlukan audit manual. Dataset tidak disertakan dalam arsip.
- Perubahan pada salinan data: split ulang, deduplikasi persis dan penamaan kelas menjadi `crack`.

## Langkah 1 — Training di Colab
1. Buka https://colab.research.google.com/ lalu upload `Wall_Crack_Detection_YOLO11.ipynb`.
2. Aktifkan GPU melalui Runtime > Change runtime type.
3. Jalankan semua sel berurutan (sel upload foto pribadi opsional).
4. Masukkan API key Roboflow saat diminta atau simpan di Secrets sebagai `ROBOFLOW_API_KEY`. Jangan unggah API key ke GitHub.
5. Jika label hilang, periksa sumber. Untuk gambar tanpa retakan yang telah diverifikasi, buat file label kosong, lalu ulangi audit. Jangan menganggap semua label hilang sebagai negatif.
6. Setelah training dan evaluasi selesai, unduh `best.pt` dan `training_report.zip`. Training belum dilakukan pada saat project dibuat.

Konfigurasi awal: 100 epochs (early stopping patience 20), batch 8, image size 640, pretrained YOLO11n. Ubah batch ke 4 jika GPU kehabisan memori. Penggunaan Colab GPU tergantung kuota akun.

## Langkah 2 — Website lokal
Gunakan Python 3.10 atau 3.11. Ekstrak project, buka terminal pada folder project, lalu pindahkan `best.pt` ke `models/best.pt`.

Windows (Command Prompt):
```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

macOS/Linux:
```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Buka http://localhost:8501 di browser. Upload gambar dinding, atur threshold dan unduh hasil prediksi.

## Struktur
```text
wall_crack_project/
├── Wall_Crack_Detection_YOLO11.ipynb
├── prepare_dataset.py
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── models/
    └── README.md  # simpan best.pt di sini
```

## Evaluasi dan portofolio
Isi metrik precision, recall, mAP50, dan mAP50-95 dari `test_metrics.json` setelah training. Tambahkan screenshot hasil deteksi foto dinding sendiri, contoh false positives/false negatives, serta keterbatasan model. Jangan mengklaim nilai akurasi sebelum evaluasi. Pertimbangkan menambahkan foto dinding tanpa retakan, noda, sambungan dan tekstur sebagai hard negatives beranotasi kosong.

Model tidak mengukur lebar retak atau kelayakan struktur. Tidak terdeteksi bukan berarti aman. Untuk penggunaan publik atau komersial, perhatikan lisensi Ultralytics (AGPL-3.0 atau Enterprise): https://www.ultralytics.com/license.

## Referensi
- https://docs.ultralytics.com/models/yolo11/
- https://docs.ultralytics.com/usage/python/
- https://roboflow.github.io/roboflow-python/

# 🎓 Proyek Akhir: Menyelesaikan Permasalahan Dropout Mahasiswa di Jaya Jaya Institut

## Business Understanding

Institut **Jaya Jaya** berfokus pada peningkatan kualitas pendidikan tinggi dengan menyediakan platform pembelajaran online. Namun, perusahaan menghadapi masalah serius dengan tingginya angka *dropout* mahasiswa yang berdampak pada penurunan kepercayaan pengguna dan potensi kerugian pendapatan.

Tingkat *dropout* yang tinggi dapat berasal dari berbagai faktor seperti performa akademik, kondisi sosial ekonomi, dan keterlibatan mahasiswa dalam pembelajaran. Oleh karena itu, diperlukan pendekatan berbasis Machine Learning untuk mengidentifikasi mahasiswa yang berisiko tinggi *dropout* sejak dini.

### Permasalahan Bisnis

- Tidak adanya sistem prediktif yang mampu mendeteksi potensi *dropout* lebih awal.
- Kesulitan dalam memahami faktor-faktor utama penyebab mahasiswa keluar dari institusi.
- Tidak tersedia dashboard interaktif untuk menyampaikan insight kepada pemangku kepentingan.

### Cakupan Proyek

- Melakukan analisis eksploratif terhadap data mahasiswa.
- Melakukan preprocessing, feature engineering, dan penanganan data imbalance menggunakan SMOTE.
- Membangun dan mengevaluasi model Machine Learning untuk prediksi *dropout*.
- Menyimpan artefak model untuk kebutuhan deployment.
- Membuat dashboard bisnis di Looker Studio.
- Membangun prototype web app interaktif menggunakan Streamlit untuk demo sistem prediksi *dropout*.

### Persiapan

**Sumber Data:**  
🔗 Berikut info link sumber dataset : https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv

**Setup environment:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
pip install streamlit
```

### Business Dashboard
Dashboard interaktif dibuat menggunakan Looker Studio yang menampilkan:
- Rasio mahasiswa aktif vs dropout.
- Visualisasi distribusi fitur penting.
- Insight faktor risiko dropout.

🔗 Akses Dashboard Looker Studio : https://lookerstudio.google.com/reporting/7df40e6e-f573-4bc5-8f04-07fdf30595ce

### Menjalankan Sistem Machine Learning
Model Machine Learning dikembangkan menggunakan pipeline lengkap dan telah disimpan sebagai artefak (dropout_model.pkl dan feature_coulumns.pkl). Sistem dapat dijalankan melalui Streamlit App.

▶️ Cara Menjalankan:
```bash
streamlit run app.py
```

🔗 Akses Streamlit App Prediksi Dropout : https://data-science-dashboard-dropout-institution.streamlit.app/

### Conclusion
#### Model prediksi dropout berhasil dibangun dan menunjukkan performa tinggi dengan akurasi yang baik. Proyek ini mampu membantu perusahaan dalam:
- Mengidentifikasi mahasiswa berisiko tinggi secara dini.
- Memberikan wawasan strategis kepada manajemen.
- Meningkatkan efektivitas intervensi akademik.

### Rekomendasi Action Items
- Integrasikan model ke dalam sistem internal perusahaan secara real-time.
- Tindak lanjuti prediksi dengan sesi bimbingan atau dukungan akademik.
- Tingkatkan kualitas data dengan menambahkan variabel perilaku dan psikologis.
- Lakukan evaluasi berkala terhadap performa model dan retraining jika diperlukan.


# -*- coding: utf-8 -*-
"""student_dropout_prediction_app.py

Aplikasi Streamlit untuk Prediksi Dropout Siswa
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# Clear cache
st.cache_resource.clear()

# Load model dan komponen
@st.cache_data
def load_artifacts():
    model, features = joblib.load('model/dropout_model.pkl')
    return model, features

model, feature_columns = load_artifacts()

# Judul aplikasi
st.title("🎓 Prediksi Risiko Dropout Mahasiswa")
st.markdown("""
Aplikasi ini memprediksi risiko dropout mahasiswa berdasarkan karakteristik akademik dan demografik.
""")

# Sidebar untuk input pengguna
with st.sidebar:
    st.header("Input Parameter Mahasiswa")
    st.markdown("Silakan isi data Mahasiswa berikut:")
    
    # Input numerik
    prev_grade = st.slider("Nilai Kualifikasi Sebelumnya", 0, 200, 120)
    academic_perf = st.slider("Rata-rata Nilai Akademik", 0, 20, 12)
    grade_diff = st.slider("Perbedaan Nilai Antar Semester", -10, 10, 0)
    age = st.slider("Usia Saat Pendaftaran", 15, 50, 20)
    
    # Input kategorikal
    marital_status = st.selectbox("Status Pernikahan", ['Single', 'Married', 'Widower', 'Divorced', 'Separated', 'Civil Union'])
    debtor = st.selectbox("Status Hutang", ['No', 'Yes'])
    tuition_fees = st.selectbox("Pembayaran Uang Kuliah Tepat Waktu", ['No', 'Yes'])
    gender = st.selectbox("Jenis Kelamin", ['Female', 'Male'])
    scholarship = st.selectbox("Penerima Beasiswa", ['No', 'Yes'])
    international = st.selectbox("Siswa Internasional", ['No', 'Yes'])
    attendance = st.selectbox("Tingkat Kehadiran", ['Partial', 'Full'])

# Fungsi untuk memproses input
def preprocess_input():
    # Buat dataframe dari input
    input_data = {
        'Marital_status': [marital_status],
        'Previous_qualification_grade': [prev_grade],
        'academic_performance': [academic_perf],
        'grade_difference': [grade_diff],
        'Debtor': [debtor],
        'Tuition_fees_up_to_date': [tuition_fees],
        'Gender': [gender],
        'Scholarship_holder': [scholarship],
        'Age_at_enrollment': [age],
        'International': [international],
        'attendance_rate': [attendance]
    }
    
    df = pd.DataFrame(input_data)
    
    # One-hot encoding untuk fitur kategorikal
    df_encoded = pd.get_dummies(df)
    
    # Pastikan semua kolom ada (untuk handling kategori yang mungkin tidak ada di input)
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Urutkan kolom sesuai dengan model
    df_encoded = df_encoded[feature_columns]
    
    return df_encoded

# Tombol prediksi
if st.button("Prediksi Risiko Dropout"):
    try:
        # Preprocess input
        input_df = preprocess_input()
        
        # Prediksi
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[0][1]  # Probabilitas dropout
        
        # Tampilkan hasil
        st.subheader("Hasil Prediksi")
        
        if prediction[0] == 1:
            st.error(f"🚨 Risiko Tinggi Dropout (Probabilitas: {proba:.1%})")
            
            # Rekomendasi
            st.markdown("""
            **Rekomendasi:**
            - Berikan bimbingan akademik intensif
            - Pantau perkembangan nilai secara berkala
            - Tawarkan konseling akademik
            """)
        else:
            st.success(f"✅ Risiko Rendah Dropout (Probabilitas: {proba:.1%})")
            st.markdown("""
            **Rekomendasi:**
            - Lanjutkan monitoring rutin
            - Pertahankan performa akademik
            """)
            
        # Visualisasi probabilitas
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.barh(['Dropout', 'Lulus/Tetap'], [proba, 1-proba], color=['red', 'green'])
        ax.set_xlim(0, 1)
        ax.set_title('Distribusi Probabilitas')
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Terjadi error: {str(e)}")

# Bagian analisis tambahan
st.markdown("---")
st.header("📊 Analisis Data Mahasiswa")

# Upload data untuk analisis batch
uploaded_file = st.file_uploader("Unggah file CSV untuk analisis batch", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        
        # Pastikan kolom sesuai
        required_cols = features = ['Marital_status', 'Previous_qualification_grade',
                                  'academic_performance', 'grade_difference',
                                  'Debtor', 'Tuition_fees_up_to_date', 'Gender',
                                  'Scholarship_holder', 'Age_at_enrollment',
                                  'International', 'attendance_rate']
        
        if all(col in batch_data.columns for col in required_cols):
            # Preprocessing
            batch_processed = pd.get_dummies(batch_data[required_cols], drop_first=True)
            
            # Pastikan kolom sesuai dengan model
            for col in feature_columns:
                if col not in batch_processed.columns:
                    batch_processed[col] = 0
            batch_processed = batch_processed[feature_columns]
            
            # Prediksi
            predictions = model.predict(batch_processed)
            probas = model.predict_proba(batch_processed)[:, 1]
            
            # Tambahkan hasil ke dataframe
            results = batch_data.copy()
            results['Predicted_Dropout_Probability'] = probas
            results['Prediction'] = np.where(probas > 0.5, 'High Risk', 'Low Risk')
            
            # Tampilkan hasil
            st.subheader("Hasil Prediksi Batch")
            st.dataframe(results.sort_values('Predicted_Dropout_Probability', ascending=False))
            
            # Download hasil
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Hasil Prediksi",
                data=csv,
                file_name='dropout_predictions.csv',
                mime='text/csv'
            )
            
            # Visualisasi distribusi
            st.subheader("Distribusi Risiko Dropout")
            fig, ax = plt.subplots()
            sns.histplot(results['Predicted_Dropout_Probability'], bins=20, kde=True, ax=ax)
            ax.set_xlabel('Probabilitas Dropout')
            ax.set_ylabel('Jumlah Mahasiswa')
            st.pyplot(fig)
            
        else:
            st.error("File CSV harus mengandung kolom yang sesuai dengan data training")
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Informasi model
with st.expander("ℹ️ Tentang Model"):
    st.markdown("""
    **Spesifikasi Model:**
    - Algoritma: Gradient Boosting Classifier
    - Metrik Evaluasi: ROC AUC Score
    - Fitur Utama:
        - Performa Akademik
        - Status Keuangan
        - Karakteristik Demografik
    """)
    
    st.markdown("""
    **Cara Penggunaan:**
    1. Isi parameter Mahasiswa di sidebar
    2. Klik tombol 'Prediksi Risiko Dropout'
    3. Lihat hasil dan rekomendasi
    """)

# Catatan kaki
st.markdown("---")
st.caption("Aplikasi Prediksi Dropout Mahasiswa - Dibangun dengan Streamlit")

# -*- coding: utf-8 -*-
"""student_dropout_prediction_app.py

Aplikasi Streamlit untuk Prediksi Dropout Siswa
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Dropout Mahasiswa", page_icon="🎓", layout="wide")

# Load model dan feature columns
@st.cache_resource
def load_model():
    model = joblib.load('model/dropout_model.pkl')
    return model

@st.cache_resource
def load_features():
    return joblib.load('model/feature_columns.pkl')

# Inisialisasi global
model = load_model()
feature_columns = load_features()


# UI Utama
st.title("🎓 Prediksi Risiko Dropout Mahasiswa")
st.markdown("Aplikasi ini memprediksi risiko dropout mahasiswa berdasarkan data akademik dan demografik.")

# Sidebar input
with st.sidebar:
    st.header("📥 Input Data Mahasiswa")
    prev_grade = st.slider("Nilai Kualifikasi Sebelumnya", 0, 200, 120)
    academic_perf = st.slider("Rata-rata Nilai Akademik", 0.0, 20.0, 12.0)
    grade_diff = st.slider("Perbedaan Nilai Antar Semester", -10.0, 10.0, 0.0)
    age = st.slider("Usia Saat Pendaftaran", 15, 50, 20)
    
    marital_status = st.selectbox("Status Pernikahan", ['Single', 'Married', 'Widower', 'Divorced', 'Separated', 'Civil Union'])
    debtor = st.selectbox("Status Hutang", ['No', 'Yes'])
    tuition_fees = st.selectbox("Pembayaran Uang Kuliah Tepat Waktu", ['No', 'Yes'])
    gender = st.selectbox("Jenis Kelamin", ['Female', 'Male'])
    scholarship = st.selectbox("Penerima Beasiswa", ['No', 'Yes'])
    international = st.selectbox("Siswa Internasional", ['No', 'Yes'])
    attendance = st.selectbox("Tingkat Kehadiran", ['Partial', 'Full'])

# Preprocessing input pengguna
def preprocess_input():
    raw = {
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
    df = pd.DataFrame(raw)
    df = pd.get_dummies(df)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]

# Tombol Prediksi
if st.button("🔍 Prediksi Dropout"):
    try:
        input_df = preprocess_input()
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.subheader("📈 Hasil Prediksi")
        if prediction == 1:
            st.error(f"🚨 Risiko **Tinggi** Dropout (Probabilitas: {proba:.1%})")
            st.markdown("- Bimbingan akademik intensif\n- Monitoring rutin\n- Konseling")
        else:
            st.success(f"✅ Risiko **Rendah** Dropout (Probabilitas: {proba:.1%})")
            st.markdown("- Pertahankan performa\n- Tetap dipantau berkala")

        # Visualisasi probabilitas
        fig, ax = plt.subplots()
        ax.barh(['Dropout', 'Tidak Dropout'], [proba, 1-proba], color=['red', 'green'])
        ax.set_xlim(0, 1)
        ax.set_title("Distribusi Probabilitas")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error saat prediksi: {e}")

# --- Upload CSV untuk analisis batch ---
st.markdown("---")
st.header("📊 Analisis Batch Mahasiswa (CSV)")

uploaded_file = st.file_uploader("Unggah file CSV", type='csv')
if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        expected_cols = ['Marital_status', 'Previous_qualification_grade',
                         'academic_performance', 'grade_difference',
                         'Debtor', 'Tuition_fees_up_to_date', 'Gender',
                         'Scholarship_holder', 'Age_at_enrollment',
                         'International', 'attendance_rate']
        
        if not all(col in batch_data.columns for col in expected_cols):
            st.error("⚠️ Kolom dalam file CSV tidak sesuai.")
        else:
            batch_encoded = pd.get_dummies(batch_data[expected_cols])
            for col in feature_columns:
                if col not in batch_encoded.columns:
                    batch_encoded[col] = 0
            batch_encoded = batch_encoded[feature_columns]

            preds = model.predict(batch_encoded)
            probas = model.predict_proba(batch_encoded)[:,1]

            batch_data['Dropout_Probability'] = probas
            batch_data['Prediction'] = np.where(preds == 1, 'High Risk', 'Low Risk')

            st.subheader("📋 Hasil Prediksi Batch")
            st.dataframe(batch_data)

            # Download hasil
            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Unduh Hasil", data=csv, file_name="batch_predictions.csv", mime='text/csv')

            # Distribusi visual
            fig2, ax2 = plt.subplots()
            sns.histplot(probas, kde=True, bins=20, ax=ax2)
            ax2.set_xlabel("Probabilitas Dropout")
            ax2.set_ylabel("Jumlah Mahasiswa")
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Gagal memproses file: {e}")

# Tentang
with st.expander("ℹ️ Tentang Aplikasi"):
    st.markdown("""
    - Dibangun menggunakan: **Streamlit**, **Scikit-learn**, dan **Python**
    - Model: Gradient Boosting Classifier
    - Output: Probabilitas dan rekomendasi risiko dropout
    """)

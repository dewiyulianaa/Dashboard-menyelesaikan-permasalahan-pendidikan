import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="wide")

# Load model
model = joblib.load("best_model.joblib")

st.title("🎓 Prediksi Risiko Dropout Mahasiswa")

# Form input
st.header("📥 Masukkan Data Mahasiswa")
with st.form("form_prediksi"):
    Gender = st.selectbox("Gender", ['0', '1'])
    Age = st.slider("Umur saat mendaftar", 17, 60, 20)
    Course = st.selectbox("Kode Course", [171, 9254, 9070, 9773, 8014, 9991, 9500, 9238])
    Admission_grade = st.number_input("Admission Grade", value=120.0)
    Debtor = st.selectbox("Status Utang", ['0', '1'])
    Scholarship_holder = st.selectbox("Penerima Beasiswa", ['0', '1'])
    Tuition_fees_up_to_date = st.selectbox("Pembayaran Lunas", ['0', '1'])
    Curricular_units_1st_sem_grade = st.number_input("Nilai Semester 1", value=12.0)
    Curricular_units_2nd_sem_grade = st.number_input("Nilai Semester 2", value=12.0)
    GDP = st.number_input("GDP", value=1.5)
    Inflation_rate = st.number_input("Inflasi", value=1.2)
    Unemployment_rate = st.number_input("Pengangguran", value=10.0)

    submitted = st.form_submit_button("🔍 Prediksi")

# Prediksi
if submitted:
    avg_sem_grade = (Curricular_units_1st_sem_grade + Curricular_units_2nd_sem_grade) / 2
    total_units_approved = 12

    input_dict = {
        'Gender': [int(Gender)],
        'Age_at_enrollment': [Age],
        'Course': [Course],
        'Admission_grade': [Admission_grade],
        'Debtor': [int(Debtor)],
        'Scholarship_holder': [int(Scholarship_holder)],
        'Tuition_fees_up_to_date': [int(Tuition_fees_up_to_date)],
        'Curricular_units_1st_sem_grade': [Curricular_units_1st_sem_grade],
        'Curricular_units_2nd_sem_grade': [Curricular_units_2nd_sem_grade],
        'avg_sem_grade': [avg_sem_grade],
        'GDP': [GDP],
        'Inflation_rate': [Inflation_rate],
        'Unemployment_rate': [Unemployment_rate],
        'total_units_approved': [total_units_approved],
    }

    # Kolom lainnya dengan nilai default
    other_columns = {
        'Marital_status': 1, 'Application_mode': 1, 'Application_order': 1,
        'Daytime_evening_attendance': 1, 'Previous_qualification': 1,
        'Previous_qualification_grade': 120.0, 'Nacionality': 1,
        'Mothers_qualification': 1, 'Fathers_qualification': 1,
        'Mothers_occupation': 1, 'Fathers_occupation': 1, 'Displaced': 0,
        'Educational_special_needs': 0, 'International': 0,
        'Curricular_units_1st_sem_credited': 6,
        'Curricular_units_1st_sem_enrolled': 6,
        'Curricular_units_1st_sem_evaluations': 6,
        'Curricular_units_1st_sem_approved': 6,
        'Curricular_units_1st_sem_without_evaluations': 0,
        'Curricular_units_2nd_sem_credited': 6,
        'Curricular_units_2nd_sem_enrolled': 6,
        'Curricular_units_2nd_sem_evaluations': 6,
        'Curricular_units_2nd_sem_approved': 6,
        'Curricular_units_2nd_sem_without_evaluations': 0,
    }

    for k, v in other_columns.items():
        input_dict[k] = [v]

    input_df = pd.DataFrame(input_dict)
    prediction = model.predict(input_df)[0]

    st.subheader("🧠 Hasil Prediksi:")
    st.success(f"Status Mahasiswa: **{prediction}**")

# Visualisasi
st.header("📊 Visualisasi Data Mahasiswa")
try:
    df = pd.read_csv("data.csv", sep=';')

    # Buat kolom tambahan jika belum ada
    if 'avg_sem_grade' not in df.columns:
        df['avg_sem_grade'] = (df['Curricular_units_1st_sem_grade'] + df['Curricular_units_2nd_sem_grade']) / 2

    if 'total_units_approved' not in df.columns:
        df['total_units_approved'] = df['Curricular_units_1st_sem_approved'] + df['Curricular_units_2nd_sem_approved']

    tab = st.selectbox("Pilih Visualisasi", ["Distribusi Status", "Rata-rata Nilai", "Boxplot"])

    if tab == "Distribusi Status":
        st.subheader("Distribusi Status Mahasiswa")
        fig1, ax1 = plt.subplots()
        df['Status'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax1)
        ax1.axis('equal')
        st.pyplot(fig1)

    elif tab == "Rata-rata Nilai":
        st.subheader("Distribusi Rata-rata Nilai Semester")
        fig2, ax2 = plt.subplots()
        sns.histplot(data=df, x='avg_sem_grade', hue='Status', kde=True, ax=ax2)
        st.pyplot(fig2)

    elif tab == "Boxplot":
        st.subheader("Boxplot Nilai dan Status")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, x='Status', y='Admission_grade', ax=ax3)
        st.pyplot(fig3)

        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, x='Status', y='total_units_approved', ax=ax4)
        st.pyplot(fig4)

except Exception as e:
    st.warning(f"Data gagal dimuat: {e}")

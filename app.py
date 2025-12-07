import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Judul aplikasi
st.title("Clustering - World Happiness Report (K-Means)")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("world_happiness_combined.csv", sep=';')

    # PERBAIKAN: Ganti koma → titik pada data numerik
    df = df.replace(",", ".", regex=True)

    # PERBAIKAN: Ubah kolom numerik menjadi float
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except:
            pass

    return df

df = load_data()

st.subheader("Dataset")
st.write(df)

# Fitur yang sesuai dataset
features = ["Happiness score", "GDP per capita", "Healthy life expectancy"]

st.subheader("Pilih variabel untuk clustering")
selected_features = st.multiselect("Pilih fitur", features, default=features)

if len(selected_features) == 0:
    st.warning("Pilih minimal 1 fitur!")
    st.stop()

# Slider jumlah cluster
k = st.slider("Jumlah cluster (k)", 2, 10, 3)

# Preprocessing (Sudah aman sekarang)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[selected_features])

# Modeling
kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_data)

# Output
st.subheader("Hasil Clustering")
st.write(df)

# Visualisasi
st.subheader("Visualisasi Cluster (2D)")
if len(selected_features) >= 2:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.scatter(df[selected_features[0]],
                df[selected_features[1]],
                c=df["Cluster"])
    plt.xlabel(selected_features[0])
    plt.ylabel(selected_features[1])
    plt.title("Visualisasi Clustering")
    st.pyplot(plt)
else:
    st.info("Pilih minimal 2 fitur untuk visualisasi 2D.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error

st.set_page_config(layout="wide")
st.title("📈 Spatial Noise Pollution Heatmap (IDW Interpolation)")

st.markdown("""
This app uses **Inverse Distance Weighting (IDW)** to create a heatmap of daytime noise pollution across Indian cities.
You can either upload your own datasets or use a **built-in dummy dataset** to test it.
""")

# ----------------------
# Define dummy data
# ----------------------

def generate_dummy_data():
    np.random.seed(42)
    city_coords = {
        'Delhi': (28.7041, 77.1025),
        'Bengaluru': (12.9716, 77.5946),
        'Chennai': (13.0827, 80.2707),
        'Hyderabad': (17.3850, 78.4867),
        'Kolkata': (22.5726, 88.3639),
        'Lucknow': (26.8467, 80.9462),
        'Mumbai': (19.0760, 72.8777),
        'Navi Mumbai': (19.0330, 73.0297)
    }

    stations = []
    noise = []

    for city, (lat, lon) in city_coords.items():
        for i in range(5):  # 5 stations per city
            station_name = f"{city}_Station_{i+1}"
            stations.append({'Station': station_name, 'City': city})
            noise.append({
                'Station': station_name,
                'Day': np.random.uniform(60, 85),
                'Night': np.random.uniform(50, 75)
            })

    stations_df = pd.DataFrame(stations)
    noise_df = pd.DataFrame(noise)
    return stations_df, noise_df, city_coords

stations_df, noise_df, city_coords = generate_dummy_data()

# ----------------------
# File upload
# ----------------------
st.subheader("Upload your data (optional)")

uploaded_stations = st.file_uploader("Upload stations.csv", type="csv")
uploaded_noise = st.file_uploader("Upload station_month.csv", type="csv")

if uploaded_stations and uploaded_noise:
    stations_df = pd.read_csv(uploaded_stations)
    noise_df = pd.read_csv(uploaded_noise)
    st.success("✅ Uploaded data loaded successfully!")
else:
    st.info("Using built-in dummy data.")

# ----------------------
# Data Preprocessing
# ----------------------
noise_df['Day'] = noise_df['Day'].fillna(noise_df['Day'].mean())
noise_df['Night'] = noise_df['Night'].fillna(noise_df['Night'].mean())

# Merge
merged_df = pd.merge(noise_df, stations_df, on='Station', how='left')
merged_df['Latitude'] = merged_df['City'].map(lambda x: city_coords.get(x, (np.nan, np.nan))[0])
merged_df['Longitude'] = merged_df['City'].map(lambda x: city_coords.get(x, (np.nan, np.nan))[1])
merged_df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

avg_noise = merged_df.groupby(['Station', 'City', 'Latitude', 'Longitude'])['Day'].mean().reset_index()

# ----------------------
# Interpolation grid
# ----------------------
x = np.linspace(72.0, 89.0, 100)
y = np.linspace(8.0, 35.0, 100)
X, Y = np.meshgrid(x, y)
points = avg_noise[['Longitude', 'Latitude']].values
values = avg_noise['Day'].values

# ----------------------
# IDW Interpolation
# ----------------------
try:
    Z = griddata(points, values, (X, Y), method='cubic')
    Z_nearest = griddata(points, values, (X, Y), method='nearest')
    Z = np.where(np.isnan(Z), Z_nearest, Z)
except Exception as e:
    st.warning("Cubic interpolation failed. Falling back to nearest.")
    Z = griddata(points, values, (X, Y), method='nearest')

# ----------------------
# LOOCV RMSE
# ----------------------
def loocv_idw(points, values):
    predictions = []
    for i in range(len(points)):
        train_points = np.delete(points, i, axis=0)
        train_values = np.delete(values, i)
        test_point = points[i]
        pred = griddata(train_points, train_values, [test_point], method='cubic')
        if np.isnan(pred):
            pred = griddata(train_points, train_values, [test_point], method='nearest')
        predictions.append(pred[0])
    mse = mean_squared_error(values, predictions)
    return np.sqrt(mse)

rmse = loocv_idw(points, values)
st.markdown(f"### 🔎 RMSE of Interpolation (LOOCV): `{rmse:.2f} dB`")

# ----------------------
# Plot Heatmap
# ----------------------
st.subheader("📊 Heatmap Output")

fig, ax = plt.subplots(figsize=(10, 6))
contour = ax.contourf(X, Y, Z, cmap='RdYlGn_r', levels=20)
plt.colorbar(contour, ax=ax, label='Noise Level (dB)')
ax.scatter(avg_noise['Longitude'], avg_noise['Latitude'], c='black', s=50, label='Stations')
for _, row in avg_noise.iterrows():
    ax.text(row['Longitude'], row['Latitude'], row['Station'], fontsize=8, ha='right')
ax.set_title('Noise Pollution Heatmap Across Indian Cities')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend()

st.pyplot(fig)

# ----------------------
# Download buttons
# ----------------------
st.subheader("📥 Download Data")

csv = avg_noise.to_csv(index=False).encode('utf-8')
st.download_button("Download Averaged Noise Data", csv, "avg_noise.csv", "text/csv")

npz_bytes = Z.astype(np.float32).tobytes()
st.download_button("Download Interpolated Heatmap (Raw Grid)", npz_bytes, "noise_heatmap_grid.raw", "application/octet-stream")
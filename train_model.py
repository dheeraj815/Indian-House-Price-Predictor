# train_model.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Paths
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'data', 'Bengaluru_House_Data.csv')

# Load dataset
df = pd.read_csv(data_path)
df = df.dropna()

# Extract BHK from size
df['bhk'] = df['size'].apply(lambda x: int(
    x.split(' ')[0]) if isinstance(x, str) else None)

# Convert sqft: handle ranges like "2100 - 2850" and strings like "1500Sq. Meter"


def convert_sqft_to_num(x):
    try:
        if '-' in x:
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None


df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
df = df[df['total_sqft'].notnull()]

# Keep relevant columns
df = df[['location', 'total_sqft', 'bath', 'bhk', 'price']]
df = df[df['bath'] <= df['bhk'] + 2]  # remove outliers

# Encode location
le = LabelEncoder()
df['location'] = le.fit_transform(df['location'])

# Split features & target
X = df[['location', 'total_sqft', 'bath', 'bhk']]
y = df['price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Save model & tools
model_dir = os.path.join(base_path, 'model')
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, 'indian_house_model.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
joblib.dump(le, os.path.join(model_dir, 'location_encoder.pkl'))

print("✅ Model, scaler & encoder saved to 'model/' folder.")

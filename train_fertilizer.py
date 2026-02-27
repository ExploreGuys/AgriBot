import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import os

# 1. Load the NEW CSV file
# Ensure the path matches where you saved the 300-row file
df = pd.read_csv('core/data/Fertilizer_Prediction_300.csv')

# 2. Clean and Normalize Data (CRITICAL for this dataset)
# This ensures "Rice", "rice ", and "RICE" are all treated as the same label
df['Crop Type'] = df['Crop Type'].str.lower().str.strip()
df['Fertilizer Name'] = df['Fertilizer Name'].str.strip()

# 3. Feature Selection
# These columns must exist exactly as named in your CSV
required_columns = ['Nitrogen', 'Phosphorous', 'Potassium', 'Crop Type', 'Fertilizer Name']
df = df[required_columns]

# 4. Initialize and Fit Encoders
crop_le = LabelEncoder()
ferti_le = LabelEncoder()

# Encode Crop names (rice, maize, chickpea, etc.)
df['Crop Type'] = crop_le.fit_transform(df['Crop Type'])
# Encode Fertilizer names (Urea, DAP, etc.)
df['Fertilizer Name'] = ferti_le.fit_transform(df['Fertilizer Name'])

# 5. Define Features (X) and Target (y)
X = df[['Nitrogen', 'Phosphorous', 'Potassium', 'Crop Type']]
y = df['Fertilizer Name']

# 6. Train the XGBoost Model
model = XGBClassifier()
model.fit(X, y)

# 7. Save the results to core/models/
MODEL_DIR = 'core/models/'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Saving models and encoders for use in Django
joblib.dump(model, os.path.join(MODEL_DIR, 'fertilizer_model.pkl'))
joblib.dump(crop_le, os.path.join(MODEL_DIR, 'ferti_crop_encoder.pkl'))
joblib.dump(ferti_le, os.path.join(MODEL_DIR, 'ferti_label_encoder.pkl'))

print("✅ Success! Trained on 4 features with 300 rows.")
print(f"Crops recognized: {len(crop_le.classes_)}")
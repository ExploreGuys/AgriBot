import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# 1. Load data
df = pd.read_csv('Crop_recommendation.csv')
X = df.drop('label', axis=1)
y = df['label']

# 2. Encode crop names
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Train model
model = XGBClassifier()
model.fit(X, y_encoded)

# 4. Save to the CORRECT folder
target_dir = 'core/models/'
os.makedirs(target_dir, exist_ok=True)

joblib.dump(model, os.path.join(target_dir, 'crop_model.pkl'))
joblib.dump(le, os.path.join(target_dir, 'label_encoder.pkl'))

print(f"✅ Actual model and encoder saved to {target_dir}")
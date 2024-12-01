# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv("recruitment_data.csv")

# Encode categorical variables if necessary
data['education'] = data['education'].map({'Bachelor\'s': 1, 'Master\'s': 2, 'PhD': 3})

# Select features (X) and target (y)
X = data[['experience', 'education', 'skills_score', 'interview_score']]
y = data['retained']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model for use in Flask
joblib.dump(model, "recruitment_model.pkl")
print("Model trained and saved as recruitment_model.pkl")

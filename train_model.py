#!/usr/bin/env python3
"""
train_model.py - Student Stress Level Prediction Model Trainer
Trains a Random Forest Classifier using scikit-learn on a calibrated student lifestyle & academic stress dataset.
Saves:
  - stress_model.pkl: Trained RandomForestClassifier
  - scaler.pkl: StandardScaler fitted on input features
  - student_data.xlsx: Seeded Excel file with headers if not present
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import openpyxl

def generate_synthetic_dataset(n_samples=3500, random_state=42):
    """
    Generates realistic student lifestyle and academic data with empirical stress correlations.
    Features:
      1. Age: 17 to 28
      2. Sleep_Hours: 3.0 to 11.0 hrs/day
      3. Study_Hours: 1.0 to 14.0 hrs/day
      4. Screen_Time: 1.5 to 14.0 hrs/day
      5. Physical_Activity: 0.0 to 18.0 hrs/week
      6. Social_Support: 1 to 5 (integer scale)
      7. GPA: 2.0 to 10.0 (0-10 scale)
    """
    np.random.seed(random_state)

    age = np.random.randint(17, 28, size=n_samples)
    sleep_hours = np.clip(np.random.normal(6.8, 1.6, size=n_samples), 3.0, 11.0)
    study_hours = np.clip(np.random.normal(5.5, 2.5, size=n_samples), 1.0, 14.0)
    screen_time = np.clip(np.random.normal(6.2, 2.4, size=n_samples), 1.5, 14.0)
    physical_activity = np.clip(np.random.exponential(3.8, size=n_samples), 0.0, 18.0)
    social_support = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.10, 0.20, 0.35, 0.25, 0.10])
    gpa = np.clip(np.random.normal(7.4, 1.5, size=n_samples), 2.0, 10.0)

    # Calculate underlying stress score based on behavioral & clinical literature
    # Higher sleep, physical activity, social support reduce stress
    # Higher screen time, excessive study (or very low study with low GPA) increase stress
    stress_index = (
        (8.0 - sleep_hours) * 1.6 +
        (screen_time - 4.0) * 0.9 +
        np.maximum(0, study_hours - 7.0) * 1.2 +
        (5.0 - social_support) * 1.5 -
        (physical_activity * 0.4) +
        np.maximum(0, 6.0 - gpa) * 1.1 +
        np.random.normal(0, 1.2, size=n_samples)
    )

    # Classify into Low, Medium, High
    # Low stress ~ bottom 32%, High stress ~ top 33%, Medium ~ middle 35%
    p_low, p_high = np.percentile(stress_index, [33, 67])

    labels = []
    for s in stress_index:
        if s < p_low:
            labels.append("Low")
        elif s > p_high:
            labels.append("High")
        else:
            labels.append("Medium")

    df = pd.DataFrame({
        "Age": age,
        "Sleep_Hours": np.round(sleep_hours, 1),
        "Study_Hours": np.round(study_hours, 1),
        "Screen_Time": np.round(screen_time, 1),
        "Physical_Activity": np.round(physical_activity, 1),
        "Social_Support": social_support,
        "GPA": np.round(gpa, 2),
        "Stress_Level": labels
    })

    return df

def train_and_save():
    print("Generating training dataset...")
    df = generate_synthetic_dataset(n_samples=4000)

    feature_cols = [
        "Age",
        "Sleep_Hours",
        "Study_Hours",
        "Screen_Time",
        "Physical_Activity",
        "Social_Support",
        "GPA"
    ]

    X = df[feature_cols]
    y = df["Stress_Level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=9,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Training Complete! Test Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    print("Saving model artifacts...")
    joblib.dump(model, "stress_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Saved stress_model.pkl and scaler.pkl")

    # Seed Excel file if not present
    excel_path = "student_data.xlsx"
    if not os.path.exists(excel_path):
        print(f"Creating initial {excel_path} with headers and sample records...")
        columns = [
            "Timestamp",
            "Name",
            "Age",
            "Sleep_Hours",
            "Study_Hours",
            "Screen_Time",
            "Physical_Activity",
            "Social_Support",
            "GPA",
            "Predicted_Stress_Level",
            "Confidence"
        ]
        sample_rows = [
            ["2026-09-20 09:15:00", "Alex Mercer", 20, 8.0, 4.5, 4.0, 5.0, 5, 8.8, "Low", "92.4%"],
            ["2026-09-21 11:30:22", "Jordan Rivera", 22, 5.0, 8.0, 7.5, 1.5, 2, 6.9, "High", "88.6%"],
            ["2026-09-21 15:45:10", "Samantha Chen", 21, 6.5, 6.0, 5.5, 3.0, 3, 7.8, "Medium", "84.1%"],
            ["2026-09-22 08:20:04", "Marcus Vance", 19, 7.5, 5.0, 4.5, 4.0, 4, 8.2, "Low", "90.7%"]
        ]
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Student Records"
        ws.append(columns)
        for row in sample_rows:
            ws.append(row)
        wb.save(excel_path)
        print(f"Saved {excel_path}")

if __name__ == "__main__":
    train_and_save()

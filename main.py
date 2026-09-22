#!/usr/bin/env python3
"""
main.py - AI-Based Student Stress Level Prediction Backend
Built with FastAPI, scikit-learn, joblib, and openpyxl.

Features:
  - POST /predict: Predicts student stress level (Low/Medium/High) and confidence using trained Random Forest model.
                   Appends submission record to 'student_data.xlsx' with thread-safe file locking.
  - GET /health: Status check and model inspection.
  - GET /history: Retrieves past predictions recorded in 'student_data.xlsx'.
  - GET /download-excel: Downloads the live student_data.xlsx file.
  - GET /analytics: Aggregated metrics and stress level trends for dashboard visualization.
  - Serves static index.html and style.css for standalone execution.
"""

import os
import sys
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any

import joblib
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Constants & Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "stress_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
EXCEL_PATH = os.path.join(BASE_DIR, "student_data.xlsx")
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")
STYLE_CSS_PATH = os.path.join(BASE_DIR, "style.css")

# Thread lock to guarantee Excel file integrity during concurrent writes
excel_lock = threading.Lock()

# Initialize FastAPI app
app = FastAPI(
    title="Student Stress Level Prediction API",
    description="ML-powered API predicting student stress levels and logging to Excel.",
    version="1.0.0"
)

# Enable CORS for HTML frontend fetch requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML Model and Scaler once at startup
model = None
scaler = None

def load_ml_artifacts():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("[INFO] Successfully loaded stress_model.pkl and scaler.pkl")
        except Exception as e:
            print(f"[ERROR] Failed to load model artifacts: {e}", file=sys.stderr)
    else:
        print("[WARNING] Model artifacts not found. Automatically triggering train_model.py...")
        try:
            import train_model
            train_model.train_and_save()
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("[INFO] Model trained and loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Auto-training failed: {e}", file=sys.stderr)

load_ml_artifacts()

# Pydantic input schema with thorough validation
class PredictionInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Student's name")
    age: int = Field(..., ge=10, le=100, description="Age in years (10-100)")
    sleep_hours: float = Field(..., ge=0.0, le=24.0, description="Sleep hours per day (0-24)")
    study_hours: float = Field(..., ge=0.0, le=24.0, description="Study hours per day (0-24)")
    screen_time: float = Field(..., ge=0.0, le=24.0, description="Screen time hours per day (0-24)")
    physical_activity: float = Field(..., ge=0.0, le=168.0, description="Physical activity hours per week (0-168)")
    social_support: int = Field(..., ge=1, le=5, description="Social support scale (1 to 5)")
    gpa: float = Field(..., ge=0.0, le=10.0, description="GPA / Academic Performance (0.0 to 10.0)")

EXCEL_COLUMNS = [
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

def init_excel_if_needed():
    """Ensures student_data.xlsx exists with styled header row."""
    if not os.path.exists(EXCEL_PATH):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Student Records"
        
        # Header styling
        header_fill = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        center_align = Alignment(horizontal="center", vertical="center")
        
        ws.append(EXCEL_COLUMNS)
        for col_num in range(1, len(EXCEL_COLUMNS) + 1):
            cell = ws.cell(row=1, column=col_num)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center_align
            
        # Adjust column widths
        col_widths = {
            "A": 22, "B": 24, "C": 8, "D": 14, "E": 14,
            "F": 14, "G": 18, "H": 16, "I": 10, "J": 22, "K": 14
        }
        for col, width in col_widths.items():
            ws.column_dimensions[col].width = width
            
        wb.save(EXCEL_PATH)
        print(f"[INFO] Created new Excel workbook at {EXCEL_PATH}")

def append_to_excel(row_data: List[Any]):
    """Thread-safe append of a new record row to student_data.xlsx."""
    with excel_lock:
        init_excel_if_needed()
        wb = openpyxl.load_workbook(EXCEL_PATH)
        ws = wb.active
        
        row_idx = ws.max_row + 1
        ws.append(row_data)
        
        # Style the new row
        stress_level = str(row_data[9])
        stress_colors = {
            "Low": "DCFCE7",     # Soft green
            "Medium": "FEF9C3",  # Soft yellow
            "High": "FEE2E2"     # Soft red
        }
        fill_color = stress_colors.get(stress_level, "FFFFFF")
        stress_fill = PatternFill(start_color=fill_color, end_color=fill_color, fill_type="solid")
        
        thin_border = Border(
            left=Side(style='thin', color='E2E8F0'),
            right=Side(style='thin', color='E2E8F0'),
            top=Side(style='thin', color='E2E8F0'),
            bottom=Side(style='thin', color='E2E8F0')
        )
        
        for col_num in range(1, len(row_data) + 1):
            cell = ws.cell(row=row_idx, column=col_num)
            cell.border = thin_border
            cell.alignment = Alignment(vertical="center", horizontal="center" if col_num not in (2,) else "left")
            if col_num == 10:  # Stress level column
                cell.fill = stress_fill
                cell.font = Font(bold=True)
                
        wb.save(EXCEL_PATH)

def read_excel_records() -> List[Dict[str, Any]]:
    """Reads all student prediction records from student_data.xlsx."""
    with excel_lock:
        if not os.path.exists(EXCEL_PATH):
            return []
        wb = openpyxl.load_workbook(EXCEL_PATH, data_only=True)
        ws = wb.active
        records = []
        rows = list(ws.iter_rows(values_only=True))
        if len(rows) <= 1:
            return []
        headers = [str(h).strip() if h is not None else f"Col{i}" for i, h in enumerate(rows[0])]
        for r in rows[1:]:
            if not any(r):
                continue
            item = {}
            for h, val in zip(headers, r):
                if isinstance(val, datetime):
                    item[h] = val.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    item[h] = val
            records.append(item)
        return records

# ----------------- API ENDPOINTS -----------------

@app.get("/health")
def health():
    """Health check endpoint confirming model status and database availability."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "excel_file_exists": os.path.exists(EXCEL_PATH),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
def predict_stress(input_data: PredictionInput):
    """
    Predicts student stress level and appends full record to student_data.xlsx.
    """
    global model, scaler
    if model is None or scaler is None:
        load_ml_artifacts()
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="ML Model or Scaler not loaded.")

    # Format features as expected by scaler
    # Feature ordering: Age, Sleep_Hours, Study_Hours, Screen_Time, Physical_Activity, Social_Support, GPA
    raw_features = np.array([[
        float(input_data.age),
        float(input_data.sleep_hours),
        float(input_data.study_hours),
        float(input_data.screen_time),
        float(input_data.physical_activity),
        float(input_data.social_support),
        float(input_data.gpa)
    ]])

    # Scale features
    scaled_features = scaler.transform(raw_features)

    # Predict class & probabilities
    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]
    classes = list(model.classes_)
    
    # Identify top confidence score
    max_prob = float(np.max(probabilities))
    confidence_str = f"{max_prob * 100:.1f}%"

    # Probability breakdown
    prob_dict = {cls_name: round(float(prob) * 100, 1) for cls_name, prob in zip(classes, probabilities)}

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare row for Excel logging
    row = [
        timestamp_str,
        input_data.name.strip(),
        int(input_data.age),
        round(float(input_data.sleep_hours), 1),
        round(float(input_data.study_hours), 1),
        round(float(input_data.screen_time), 1),
        round(float(input_data.physical_activity), 1),
        int(input_data.social_support),
        round(float(input_data.gpa), 2),
        str(prediction),
        confidence_str
    ]

    try:
        append_to_excel(row)
    except Exception as e:
        print(f"[ERROR] Failed appending to Excel: {e}", file=sys.stderr)
        # We still return prediction even if file write has non-fatal glitch
        return {
            "stress_level": str(prediction),
            "confidence": confidence_str,
            "confidence_score": round(max_prob * 100, 1),
            "probabilities": prob_dict,
            "timestamp": timestamp_str,
            "warning": f"Could not write to Excel: {str(e)}"
        }

    return {
        "stress_level": str(prediction),
        "confidence": confidence_str,
        "confidence_score": round(max_prob * 100, 1),
        "probabilities": prob_dict,
        "timestamp": timestamp_str,
        "excel_saved": True
    }

@app.get("/history")
def get_history():
    """Returns past prediction records saved in student_data.xlsx."""
    records = read_excel_records()
    # Reverse so latest predictions appear first
    return {"records": list(reversed(records)), "total_count": len(records)}

@app.get("/download-excel")
def download_excel():
    """Provides student_data.xlsx file download."""
    init_excel_if_needed()
    if not os.path.exists(EXCEL_PATH):
        raise HTTPException(status_code=404, detail="Excel file not found.")
    return FileResponse(
        EXCEL_PATH,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="student_data.xlsx"
    )

@app.get("/analytics")
def get_analytics():
    """Computes distribution metrics and trends for the dashboard."""
    records = read_excel_records()
    total = len(records)
    if total == 0:
        return {
            "total_students": 0,
            "distribution": {"Low": 0, "Medium": 0, "High": 0},
            "averages": {"sleep": 0, "study": 0, "screen": 0, "physical": 0, "gpa": 0}
        }
        
    counts = {"Low": 0, "Medium": 0, "High": 0}
    sum_sleep = 0.0
    sum_study = 0.0
    sum_screen = 0.0
    sum_phys = 0.0
    sum_gpa = 0.0
    
    for r in records:
        lvl = r.get("Predicted_Stress_Level", "Medium")
        counts[lvl] = counts.get(lvl, 0) + 1
        try:
            sum_sleep += float(r.get("Sleep_Hours", 0))
            sum_study += float(r.get("Study_Hours", 0))
            sum_screen += float(r.get("Screen_Time", 0))
            sum_phys += float(r.get("Physical_Activity", 0))
            sum_gpa += float(r.get("GPA", 0))
        except (ValueError, TypeError):
            pass

    return {
        "total_students": total,
        "distribution": counts,
        "averages": {
            "sleep": round(sum_sleep / total, 1),
            "study": round(sum_study / total, 1),
            "screen": round(sum_screen / total, 1),
            "physical": round(sum_phys / total, 1),
            "gpa": round(sum_gpa / total, 2)
        }
    }

# Static file serving for standalone execution
@app.get("/")
def serve_index():
    if os.path.exists(INDEX_HTML_PATH):
        return FileResponse(INDEX_HTML_PATH)
    return {"message": "Welcome to Student Stress Level Prediction API. Open index.html to view UI."}

@app.get("/style.css")
def serve_css():
    if os.path.exists(STYLE_CSS_PATH):
        return FileResponse(STYLE_CSS_PATH, media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found.")

if __name__ == "__main__":
    import uvicorn
    port = 5000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    print(f"[INFO] Starting FastAPI server on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

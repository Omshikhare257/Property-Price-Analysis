"""
PropertyIQ — Single-File App (Python Backend + JavaScript Frontend)
Run:  pip install flask pandas numpy scikit-learn plotly openpyxl pdfplumber python-pptx pillow
      python propertyiq_app.py
Then open: http://localhost:5000
"""

import io
import os
import json
import base64
import warnings
import threading
import webbrowser
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FILE PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

def parse_excel(file_bytes, filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    engine = "xlrd" if ext == "xls" else "openpyxl"
    xf = pd.ExcelFile(io.BytesIO(file_bytes), engine=engine)
    frames = []
    for sheet in xf.sheet_names:
        try:
            df = xf.parse(sheet)
            if len(df) > 0:
                frames.append(df)
        except Exception:
            pass
    if not frames:
        raise ValueError("No readable sheets found in Excel file.")
    return max(frames, key=len)

def parse_pdf(file_bytes):
    import pdfplumber
    frames = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if table and len(table) > 1:
                    try:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        frames.append(df)
                    except Exception:
                        pass
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        return combined.replace("", np.nan).dropna(how="all")
    raise ValueError("No tables found in PDF.")

def parse_pptx(file_bytes):
    from pptx import Presentation
    frames = []
    prs = Presentation(io.BytesIO(file_bytes))
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_table:
                table = shape.table
                rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
                if len(rows) > 1:
                    try:
                        df = pd.DataFrame(rows[1:], columns=rows[0])
                        frames.append(df)
                    except Exception:
                        pass
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        return combined.replace("", np.nan).dropna(how="all")
    raise ValueError("No tables found in PPTX file.")

def parse_image(file_bytes):
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(img)
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            raise ValueError("Could not extract tabular data from image.")
        sep = "," if "," in lines[0] else "\t"
        data = [l.split(sep) for l in lines]
        df = pd.DataFrame(data[1:], columns=data[0])
        return df.replace("", np.nan).dropna(how="all")
    except ImportError:
        raise ValueError("pytesseract not available.")

def load_file(file_bytes, filename):
    fn = filename.lower()
    if fn.endswith(".csv"):
        return parse_csv(file_bytes)
    elif fn.endswith((".xlsx", ".xls", ".xlsm")):
        return parse_excel(file_bytes, fn)
    elif fn.endswith(".pdf"):
        return parse_pdf(file_bytes)
    elif fn.endswith(".pptx"):
        return parse_pptx(file_bytes)
    elif fn.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return parse_image(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: '{filename}'")

# ─────────────────────────────────────────────────────────────────────────────
# CORE ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

BEDROOM_TO_NUM = {"1 RK": 0, "1 BHK": 1, "2 BHK": 2, "3 BHK": 3, "4 BHK": 4, "5 BHK": 5}
NUM_TO_BEDROOM = {v: k for k, v in BEDROOM_TO_NUM.items()}

def bedroom_to_str(x):
    if pd.isna(x): return np.nan
    try:
        xi = int(float(x))
        return NUM_TO_BEDROOM.get(xi, f"{xi} BHK")
    except: return x

def bedroom_to_num(x):
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        if x in BEDROOM_TO_NUM: return BEDROOM_TO_NUM[x]
        parts = x.split()
        if parts and parts[0].isdigit(): return int(parts[0])
    try: return int(float(x))
    except: return np.nan

class PropertyAnalyzer:
    def __init__(self):
        self.model = None
        self.model_type = None
        self.scaler = StandardScaler()
        self.features = []
        self.year_model = None
        self.location_stats = {}
        self.df = None
        self.df_processed = None

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        numeric_cols = ["price","sqft","bedrooms","bathrooms","floor","total_floors",
                        "age_years","parking","metro_distance_km","price_per_sqft"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "bedrooms" in df.columns:
            df["bedrooms"] = df["bedrooms"].apply(bedroom_to_str)
        if "age_years" in df.columns and "year" not in df.columns:
            current = pd.Timestamp.now().year
            df["year"] = (current - df["age_years"]).where(df["age_years"].notna())
        df = df.dropna(subset=["price"])
        for grp_col in ["location", "state"]:
            if grp_col in df.columns:
                agg = df.groupby(grp_col).agg(
                    mean_price=("price","mean"), median_price=("price","median"),
                    min_price=("price","min"), max_price=("price","max"),
                    count=("price","count")).round(2)
                if "price_per_sqft" in df.columns:
                    agg["avg_ppsqft"] = df.groupby(grp_col)["price_per_sqft"].mean().round(2)
                self.location_stats[grp_col] = agg
        if "year" in df.columns:
            valid = df[["year","price"]].dropna()
            if len(valid) > 5:
                self.year_model = LinearRegression()
                self.year_model.fit(valid[["year"]], valid["price"])
        self.df_processed = df
        return df

    def predict_year_price(self, year: int) -> float:
        if self.year_model is None: raise ValueError("Year model not trained.")
        return float(self.year_model.predict([[year]])[0])

    def train_model(self, model_type: str = "Random Forest") -> dict:
        df = self.df_processed.copy()
        if "bedrooms" in df.columns:
            df["bedrooms"] = df["bedrooms"].apply(bedroom_to_num)
        numeric_features = [c for c in ["sqft","bedrooms","bathrooms","floor","total_floors",
                                         "age_years","parking","metro_distance_km"] if c in df.columns]
        cat_features = [c for c in ["state","location","furnishing","facing"] if c in df.columns]
        if not numeric_features: raise ValueError("No numeric features found.")
        df = df[numeric_features + cat_features + ["price"]].dropna(subset=["price"])
        if len(df) < 10: raise ValueError(f"Only {len(df)} rows — need at least 10.")
        X = df[numeric_features].copy()
        if cat_features:
            dummies = pd.get_dummies(df[cat_features], drop_first=True)
            X = pd.concat([X, dummies], axis=1)
        y = df["price"]
        self.features = X.columns.tolist()
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        self.model = models[model_type]
        self.model.fit(X_train, y_train)
        self.model_type = model_type
        y_pred = self.model.predict(X_test)
        return {"r2": r2_score(y_test, y_pred), "mae": mean_absolute_error(y_test, y_pred),
                "n_train": len(X_train), "n_test": len(X_test)}

    def predict(self, input_dict: dict) -> float:
        if self.model is None: raise ValueError("Model not trained.")
        df_m = self.df_processed.copy()
        if "bedrooms" in df_m.columns:
            df_m["bedrooms"] = df_m["bedrooms"].apply(bedroom_to_num)
        cat_features = [c for c in ["state","location","furnishing","facing"] if c in df_m.columns]
        row_df = pd.DataFrame([input_dict])
        if cat_features:
            dummies = pd.get_dummies(row_df, columns=cat_features, drop_first=True)
        else:
            dummies = row_df.copy()
        for col in self.features:
            if col not in dummies.columns: dummies[col] = 0
        dummies = dummies[self.features]
        scaled = self.scaler.transform(dummies)
        return float(self.model.predict(scaled)[0])

# Global state
analyzer = PropertyAnalyzer()

# ─────────────────────────────────────────────────────────────────────────────
# HTML + JS FRONTEND (single-file, fully self-contained)
# ─────────────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>PropertyIQ — Price Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap" rel="stylesheet"/>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root{
    --bg:#080d18;--surface:#0f1829;--surface2:#162034;--surface3:#1d2a44;
    --accent:#f0a500;--accent2:#e05c5c;--accent3:#c084fc;
    --text:#e8eaf0;--muted:#6b7a92;--border:rgba(255,255,255,0.07);
    --radius:14px;--glow:0 0 30px rgba(240,165,0,0.12);
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  html,body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;min-height:100vh;overflow-x:hidden;}

  /* Animated grid background */
  body::before{
    content:'';position:fixed;inset:0;
    background-image:linear-gradient(rgba(240,165,0,0.03) 1px,transparent 1px),
                     linear-gradient(90deg,rgba(240,165,0,0.03) 1px,transparent 1px);
    background-size:60px 60px;animation:gridMove 20s linear infinite;z-index:0;pointer-events:none;
  }
  @keyframes gridMove{0%{background-position:0 0;}100%{background-position:60px 60px;}}

  /* Orbs */
  .orb{position:fixed;border-radius:50%;filter:blur(120px);z-index:0;pointer-events:none;animation:orbFloat 8s ease-in-out infinite;}
  .orb1{width:500px;height:500px;background:radial-gradient(circle,rgba(240,165,0,0.08),transparent 70%);top:-150px;right:-100px;}
  .orb2{width:400px;height:400px;background:radial-gradient(circle,rgba(224,92,92,0.07),transparent 70%);bottom:-100px;left:-100px;animation-delay:-4s;}

  /* Layout */
  #app{position:relative;z-index:1;max-width:1400px;margin:0 auto;padding:2rem 1.5rem;}

  /* Hero */
  .hero{padding:3rem 0 2rem;}
  .hero-badge{display:inline-flex;align-items:center;gap:0.5rem;background:rgba(240,165,0,0.1);border:1px solid rgba(240,165,0,0.25);border-radius:999px;padding:0.35rem 1rem;font-size:0.78rem;color:var(--accent);margin-bottom:1.5rem;letter-spacing:0.05em;text-transform:uppercase;}
  .hero-badge .dot{width:6px;height:6px;background:var(--accent);border-radius:50%;animation:pulse 2s infinite;}
  @keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:0.5;transform:scale(0.8);}}
  .hero-title{font-family:'Syne',sans-serif;font-size:clamp(2.5rem,5vw,4rem);font-weight:800;line-height:1.05;margin-bottom:0.75rem;}
  .hero-title span{background:linear-gradient(135deg,#f0a500 0%,#e05c5c 50%,#c084fc 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
  .hero-sub{color:var(--muted);font-size:1.05rem;max-width:600px;}

  /* Upload zone */
  .upload-zone{
    border:2px dashed rgba(240,165,0,0.3);border-radius:var(--radius);
    background:linear-gradient(135deg,rgba(15,24,41,0.8),rgba(22,32,52,0.8));
    padding:3rem 2rem;text-align:center;cursor:pointer;transition:all 0.3s;
    margin-bottom:2rem;backdrop-filter:blur(10px);
  }
  .upload-zone:hover,.upload-zone.drag-over{border-color:var(--accent);background:rgba(240,165,0,0.05);transform:translateY(-2px);box-shadow:var(--glow);}
  .upload-icon{font-size:3rem;margin-bottom:1rem;display:block;}
  .upload-title{font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:0.5rem;}
  .upload-sub{color:var(--muted);font-size:0.9rem;}
  .upload-formats{display:flex;justify-content:center;gap:0.5rem;flex-wrap:wrap;margin-top:1.2rem;}
  .fmt-tag{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:0.25rem 0.75rem;font-size:0.75rem;color:var(--muted);}
  #fileInput{display:none;}

  /* Progress bar */
  .progress-wrap{display:none;margin-bottom:1.5rem;}
  .progress-bar{height:4px;background:var(--surface2);border-radius:2px;overflow:hidden;}
  .progress-fill{height:100%;width:0;background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:2px;transition:width 0.4s;animation:shimmer 1.5s infinite linear;}
  @keyframes shimmer{0%{filter:brightness(1);}50%{filter:brightness(1.4);}100%{filter:brightness(1);}}
  .progress-label{color:var(--muted);font-size:0.85rem;margin-bottom:0.5rem;}

  /* Alert */
  .alert{padding:1rem 1.5rem;border-radius:var(--radius);margin-bottom:1.5rem;display:none;animation:fadeIn 0.3s;}
  .alert-success{background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.25);color:#34d399;}
  .alert-error{background:rgba(224,92,92,0.08);border:1px solid rgba(224,92,92,0.25);color:#e05c5c;}
  .alert-info{background:rgba(240,165,0,0.08);border:1px solid rgba(240,165,0,0.25);color:var(--accent);}
  @keyframes fadeIn{from{opacity:0;transform:translateY(-8px);}to{opacity:1;transform:translateY(0);}}

  /* KPI cards */
  .kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-bottom:2rem;}
  .kpi-card{
    background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
    padding:1.5rem;text-align:center;transition:all 0.3s;position:relative;overflow:hidden;
  }
  .kpi-card::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(240,165,0,0.03),transparent);opacity:0;transition:opacity 0.3s;}
  .kpi-card:hover{transform:translateY(-4px);border-color:rgba(240,165,0,0.3);box-shadow:var(--glow);}
  .kpi-card:hover::before{opacity:1;}
  .kpi-label{font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--muted);margin-bottom:0.6rem;}
  .kpi-value{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:var(--accent);}
  .kpi-sub{font-size:0.78rem;color:var(--muted);margin-top:0.25rem;}

  /* Tabs */
  .tabs{display:flex;gap:0.35rem;background:var(--surface);border-radius:var(--radius);padding:6px;margin-bottom:2rem;flex-wrap:wrap;}
  .tab-btn{
    flex:1;min-width:100px;padding:0.6rem 1.2rem;border:none;background:transparent;
    color:var(--muted);font-family:'Syne',sans-serif;font-weight:600;font-size:0.88rem;
    border-radius:10px;cursor:pointer;transition:all 0.25s;white-space:nowrap;
  }
  .tab-btn.active{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#0b0f1a;}
  .tab-btn:hover:not(.active){color:var(--text);background:var(--surface2);}
  .tab-panel{display:none;animation:fadeIn 0.3s;}
  .tab-panel.active{display:block;}

  /* Section header */
  .section-hdr{
    font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
    border-left:3px solid var(--accent);padding-left:0.75rem;
    margin:2rem 0 1rem;
  }

  /* Cards & panels */
  .panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.5rem;margin-bottom:1.5rem;}
  .panel-row{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;}
  @media(max-width:700px){.panel-row{grid-template-columns:1fr;}}

  /* Sidebar + main layout */
  .main-layout{display:grid;grid-template-columns:260px 1fr;gap:1.5rem;align-items:start;}
  @media(max-width:900px){.main-layout{grid-template-columns:1fr;}}
  .sidebar{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.5rem;position:sticky;top:1rem;}
  .sidebar-title{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:var(--accent);margin-bottom:1.2rem;display:flex;align-items:center;gap:0.5rem;}

  /* Form controls */
  label.ctrl-label{display:block;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.07em;color:var(--muted);margin-bottom:0.4rem;margin-top:1rem;}
  select,input[type=number],input[type=text],input[type=range]{
    width:100%;background:var(--surface2);border:1px solid var(--border);
    border-radius:8px;padding:0.55rem 0.85rem;color:var(--text);font-family:'DM Sans',sans-serif;
    font-size:0.9rem;outline:none;transition:border-color 0.2s;
    -webkit-appearance:none;appearance:none;
  }
  select{background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%236b7a92' d='M6 8L1 3h10z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 0.75rem center;}
  select:focus,input:focus{border-color:var(--accent);}
  input[type=range]{padding:0;height:6px;border-radius:3px;cursor:pointer;accent-color:var(--accent);}
  .range-labels{display:flex;justify-content:space-between;font-size:0.75rem;color:var(--muted);margin-top:0.3rem;}

  /* Buttons */
  .btn{
    display:inline-flex;align-items:center;gap:0.5rem;padding:0.65rem 1.5rem;
    border:none;border-radius:8px;font-family:'Syne',sans-serif;font-weight:700;
    font-size:0.9rem;cursor:pointer;transition:all 0.2s;letter-spacing:0.02em;
  }
  .btn-primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#0b0f1a;}
  .btn-primary:hover{opacity:0.88;transform:translateY(-2px);box-shadow:0 8px 24px rgba(240,165,0,0.25);}
  .btn-secondary{background:var(--surface2);border:1px solid var(--border);color:var(--text);}
  .btn-secondary:hover{border-color:var(--accent);color:var(--accent);}
  .btn:disabled{opacity:0.4;cursor:not-allowed;transform:none!important;}

  /* Stats pill */
  .count-pill{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:0.85rem 1rem;text-align:center;margin-top:1.2rem;}
  .count-num{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:var(--accent);}
  .count-label{font-size:0.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.07em;}

  /* Table */
  .table-wrap{overflow-x:auto;border-radius:var(--radius);border:1px solid var(--border);}
  table{width:100%;border-collapse:collapse;font-size:0.85rem;}
  thead tr{background:var(--surface2);}
  th{padding:0.85rem 1rem;text-align:left;font-family:'Syne',sans-serif;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.07em;color:var(--muted);white-space:nowrap;}
  td{padding:0.75rem 1rem;border-top:1px solid var(--border);white-space:nowrap;}
  tbody tr:hover{background:rgba(240,165,0,0.03);}
  .price-cell{color:var(--accent);font-family:'Syne',sans-serif;font-weight:700;}
  .badge{display:inline-block;padding:0.2rem 0.6rem;border-radius:999px;font-size:0.72rem;font-weight:600;}
  .badge-furnished{background:rgba(52,211,153,0.12);color:#34d399;}
  .badge-semi{background:rgba(240,165,0,0.12);color:var(--accent);}
  .badge-unfurnished{background:rgba(107,122,146,0.15);color:var(--muted);}

  /* Search */
  .search-wrap{position:relative;margin-bottom:1rem;}
  .search-wrap input{padding-left:2.5rem;}
  .search-icon{position:absolute;left:0.85rem;top:50%;transform:translateY(-50%);color:var(--muted);font-size:1rem;}

  /* Chart container */
  .chart-box{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1rem;margin-bottom:1.5rem;}

  /* Prediction result */
  .pred-result{
    background:linear-gradient(135deg,var(--surface),var(--surface2));
    border:1px solid rgba(240,165,0,0.25);border-radius:20px;
    padding:2.5rem;text-align:center;margin-top:1.5rem;
    animation:fadeIn 0.5s;position:relative;overflow:hidden;
  }
  .pred-result::before{
    content:'';position:absolute;inset:-50%;
    background:radial-gradient(circle at center,rgba(240,165,0,0.06),transparent 60%);
    animation:rotateBg 10s linear infinite;
  }
  @keyframes rotateBg{from{transform:rotate(0deg);}to{transform:rotate(360deg);}}
  .pred-label{color:var(--muted);font-size:0.8rem;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.5rem;position:relative;}
  .pred-price{font-family:'Syne',sans-serif;font-size:3.5rem;font-weight:800;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1;position:relative;}
  .pred-algo{color:var(--muted);font-size:0.8rem;margin-top:0.5rem;position:relative;}

  /* Training metrics */
  .metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.5rem;}
  .metric-box{background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:1.2rem;text-align:center;}
  .metric-val{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:var(--accent);}
  .metric-lbl{font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:var(--muted);margin-top:0.3rem;}

  /* Form grid */
  .form-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1.2rem;}

  /* Year forecast */
  .forecast-box{background:var(--surface2);border:1px solid rgba(240,165,0,0.2);border-radius:12px;padding:1.5rem;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;margin-top:1rem;}
  .forecast-val{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:var(--accent);}

  /* Pagination */
  .pagination{display:flex;align-items:center;gap:0.5rem;margin-top:1rem;justify-content:flex-end;}
  .page-btn{padding:0.4rem 0.85rem;border:1px solid var(--border);background:var(--surface2);border-radius:6px;color:var(--muted);cursor:pointer;font-size:0.85rem;transition:all 0.2s;}
  .page-btn:hover,.page-btn.active{border-color:var(--accent);color:var(--accent);}
  .page-info{font-size:0.82rem;color:var(--muted);}

  /* Download btn */
  .dl-btn{text-decoration:none;display:inline-flex;align-items:center;gap:0.5rem;}

  /* Loading skeleton */
  .skeleton{background:linear-gradient(90deg,var(--surface2) 25%,var(--surface3) 50%,var(--surface2) 75%);background-size:200% 100%;animation:skeleton 1.5s infinite;border-radius:8px;height:20px;}
  @keyframes skeleton{0%{background-position:200% 0;}100%{background-position:-200% 0;}}

  /* Year forecast row */
  .year-row{display:flex;align-items:flex-end;gap:1rem;flex-wrap:wrap;}
  .year-row .ctrl-group{flex:1;min-width:140px;}

  /* Responsive chart heights */
  .plotly-chart{width:100%;min-height:300px;}
  
  /* Scrollbar */
  ::-webkit-scrollbar{width:6px;height:6px;}
  ::-webkit-scrollbar-track{background:var(--bg);}
  ::-webkit-scrollbar-thumb{background:var(--surface2);border-radius:3px;}

  /* Location stats table */
  .loc-stat-chip{display:inline-block;background:rgba(240,165,0,0.1);border-radius:6px;padding:0.15rem 0.5rem;font-size:0.78rem;color:var(--accent);}

  /* Hidden */
  .hidden{display:none!important;}
</style>
</head>
<body>
<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div id="app">

  <!-- HERO -->
  <div class="hero">
    <div class="hero-badge"><span class="dot"></span> AI-Powered · Real-Time Analysis</div>
    <h1 class="hero-title">🏙️ Property<span>IQ</span></h1>
    <p class="hero-sub">Upload your property dataset and unlock AI-powered price intelligence, trend analysis, and market predictions.</p>
  </div>

  <!-- UPLOAD -->
  <div class="upload-zone" id="uploadZone">
    <input type="file" id="fileInput" accept=".csv,.xlsx,.xls,.xlsm,.pdf,.pptx,.jpg,.jpeg,.png"/>
    <span class="upload-icon">📂</span>
    <div class="upload-title">Drop your dataset here or click to browse</div>
    <div class="upload-sub">Supports all major formats — we'll handle the rest</div>
    <div class="upload-formats">
      <span class="fmt-tag">📄 CSV</span>
      <span class="fmt-tag">📊 Excel</span>
      <span class="fmt-tag">📑 PDF</span>
      <span class="fmt-tag">📽️ PPTX</span>
      <span class="fmt-tag">🖼️ JPG/PNG</span>
    </div>
  </div>

  <!-- PROGRESS -->
  <div class="progress-wrap" id="progressWrap">
    <div class="progress-label" id="progressLabel">Uploading…</div>
    <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
  </div>

  <!-- ALERTS -->
  <div class="alert alert-success" id="alertSuccess"></div>
  <div class="alert alert-error" id="alertError"></div>

  <!-- MAIN (hidden until data loaded) -->
  <div id="mainContent" class="hidden">
    <div class="main-layout">

      <!-- SIDEBAR FILTERS -->
      <div class="sidebar">
        <div class="sidebar-title">🎛️ Filters</div>

        <label class="ctrl-label">State</label>
        <select id="filterState"><option value="">All</option></select>

        <label class="ctrl-label">Location</label>
        <select id="filterLocation"><option value="">All</option></select>

        <label class="ctrl-label">Enterprise / Builder</label>
        <select id="filterEnterprise"><option value="">All</option></select>

        <label class="ctrl-label">Price Range (₹L)</label>
        <input type="range" id="priceMin" min="0" max="10000" step="1" value="0" oninput="updateRangeLabel()"/>
        <input type="range" id="priceMax" min="0" max="10000" step="1" value="10000" oninput="updateRangeLabel()"/>
        <div class="range-labels"><span id="priceMinLabel">₹0L</span><span id="priceMaxLabel">₹10000L</span></div>

        <label class="ctrl-label">Bedrooms</label>
        <select id="filterBedrooms"><option value="">All</option></select>

        <label class="ctrl-label">Furnishing</label>
        <select id="filterFurnishing"><option value="">All</option></select>

        <div class="count-pill">
          <div class="count-num" id="filterCount">—</div>
          <div class="count-label">Properties Shown</div>
        </div>

        <div style="margin-top:1rem;">
          <button class="btn btn-primary" style="width:100%" onclick="applyFilters()">Apply Filters</button>
        </div>
        <div style="margin-top:0.6rem;">
          <button class="btn btn-secondary" style="width:100%" onclick="resetFilters()">Reset</button>
        </div>
      </div>

      <!-- RIGHT CONTENT -->
      <div>
        <!-- TABS -->
        <div class="tabs">
          <button class="tab-btn active" onclick="switchTab('overview')">📊 Overview</button>
          <button class="tab-btn" onclick="switchTab('charts')">📈 Charts</button>
          <button class="tab-btn" onclick="switchTab('listings')">🏠 Listings</button>
          <button class="tab-btn" onclick="switchTab('predict')">🎯 Price Prediction</button>
        </div>

        <!-- ═══ OVERVIEW TAB ═══ -->
        <div id="tab-overview" class="tab-panel active">
          <div class="kpi-grid" id="kpiGrid"></div>

          <div id="yearForecastSection" class="panel hidden">
            <div class="section-hdr">📅 Year Price Forecast</div>
            <div class="year-row">
              <div class="ctrl-group">
                <label class="ctrl-label">Forecast Year</label>
                <input type="number" id="forecastYear" value="2027" min="1990" max="2060"/>
              </div>
              <button class="btn btn-primary" onclick="runYearForecast()" style="margin-bottom:0.05rem">📈 Forecast</button>
            </div>
            <div class="forecast-box hidden" id="forecastResult">
              <div>
                <div style="color:var(--muted);font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em">Predicted Avg Price for <span id="forecastYearLabel"></span></div>
                <div class="forecast-val" id="forecastVal"></div>
              </div>
              <div style="color:var(--muted);font-size:0.85rem">Based on historical trend</div>
            </div>
          </div>

          <div id="locationSnapshotSection" class="panel hidden">
            <div class="section-hdr">📍 Location Snapshot</div>
            <div class="table-wrap" id="locationTable"></div>
          </div>
        </div>

        <!-- ═══ CHARTS TAB ═══ -->
        <div id="tab-charts" class="tab-panel">
          <div class="chart-box">
            <div class="section-hdr">Price Distribution</div>
            <div id="chartHist" class="plotly-chart"></div>
          </div>
          <div class="chart-box" id="chartYearWrap" style="display:none">
            <div class="section-hdr">Year-wise Avg Price</div>
            <div id="chartYear" class="plotly-chart"></div>
          </div>
          <div class="chart-box" id="chartScatterWrap" style="display:none">
            <div class="section-hdr">Price vs Area</div>
            <div id="chartScatter" class="plotly-chart"></div>
          </div>
          <div class="chart-box" id="chartLocWrap" style="display:none">
            <div class="section-hdr">Avg Price by Location</div>
            <div id="chartLoc" class="plotly-chart"></div>
          </div>
          <div class="chart-box" id="chartFurnWrap" style="display:none">
            <div class="section-hdr">Furnishing vs Price</div>
            <div id="chartFurn" class="plotly-chart"></div>
          </div>
        </div>

        <!-- ═══ LISTINGS TAB ═══ -->
        <div id="tab-listings" class="tab-panel">
          <div class="section-hdr">🏠 Property Listings</div>
          <div class="search-wrap">
            <span class="search-icon">🔍</span>
            <input type="text" id="listingSearch" placeholder="Search location, BHK, furnishing…" oninput="renderListings()"/>
          </div>
          <div class="table-wrap" id="listingsTable"></div>
          <div class="pagination" id="paginationBar"></div>
          <div style="margin-top:1rem">
            <a id="downloadBtn" class="btn btn-secondary dl-btn" href="#" onclick="downloadCSV()">⬇️ Download Filtered CSV</a>
          </div>
        </div>

        <!-- ═══ PREDICT TAB ═══ -->
        <div id="tab-predict" class="tab-panel">
          <div class="panel">
            <div class="section-hdr">🤖 Train Prediction Model</div>
            <div style="display:flex;align-items:flex-end;gap:1rem;flex-wrap:wrap">
              <div style="flex:1;min-width:200px">
                <label class="ctrl-label">Algorithm</label>
                <select id="modelType">
                  <option value="Random Forest">Random Forest</option>
                  <option value="Gradient Boosting">Gradient Boosting</option>
                  <option value="Linear Regression">Linear Regression</option>
                </select>
              </div>
              <button class="btn btn-primary" id="trainBtn" onclick="trainModel()">🚀 Train Model</button>
            </div>

            <div id="trainResult" class="hidden" style="margin-top:1.5rem">
              <div class="metric-grid">
                <div class="metric-box"><div class="metric-val" id="r2Score">—</div><div class="metric-lbl">R² Score</div></div>
                <div class="metric-box"><div class="metric-val" id="maeScore">—</div><div class="metric-lbl">Mean Abs Error</div></div>
                <div class="metric-box"><div class="metric-val" id="nTrain">—</div><div class="metric-lbl">Training Samples</div></div>
              </div>
            </div>
          </div>

          <div id="estimateForm" class="panel hidden">
            <div class="section-hdr">💡 Get a Price Estimate</div>
            <div class="form-grid">
              <div>
                <label class="ctrl-label">Area (sqft)</label>
                <input type="number" id="fSqft" value="1000" min="100"/>
              </div>
              <div>
                <label class="ctrl-label">Bedrooms</label>
                <select id="fBedrooms">
                  <option>1 RK</option><option selected>1 BHK</option>
                  <option>2 BHK</option><option>3 BHK</option>
                  <option>4 BHK</option><option>5 BHK</option>
                </select>
              </div>
              <div>
                <label class="ctrl-label">Bathrooms</label>
                <input type="number" id="fBathrooms" value="2" min="1"/>
              </div>
              <div>
                <label class="ctrl-label">Floor</label>
                <input type="number" id="fFloor" value="5" min="1"/>
              </div>
              <div>
                <label class="ctrl-label">Total Floors</label>
                <input type="number" id="fTotalFloors" value="10" min="1"/>
              </div>
              <div>
                <label class="ctrl-label">Parking Spots</label>
                <input type="number" id="fParking" value="1" min="0"/>
              </div>
              <div>
                <label class="ctrl-label">Age (years)</label>
                <input type="number" id="fAge" value="5" min="0"/>
              </div>
              <div>
                <label class="ctrl-label">Metro Distance (km)</label>
                <input type="number" id="fMetro" value="2.0" min="0" step="0.1"/>
              </div>
              <div id="fFurnishWrap" class="hidden">
                <label class="ctrl-label">Furnishing</label>
                <select id="fFurnishing"></select>
              </div>
              <div id="fFacingWrap" class="hidden">
                <label class="ctrl-label">Facing</label>
                <select id="fFacing"></select>
              </div>
              <div id="fLocationWrap" class="hidden">
                <label class="ctrl-label">Location</label>
                <select id="fLocation"></select>
              </div>
            </div>
            <div style="margin-top:1.5rem">
              <button class="btn btn-primary" onclick="estimatePrice()" style="padding:0.75rem 2rem;font-size:1rem">💡 Estimate Price</button>
            </div>
            <div id="predResult" class="hidden pred-result">
              <div class="pred-label">Estimated Market Price</div>
              <div class="pred-price" id="predPrice"></div>
              <div class="pred-algo" id="predAlgo"></div>
            </div>
          </div>
        </div>

      </div><!-- /right content -->
    </div><!-- /main-layout -->
  </div><!-- /mainContent -->

</div><!-- /app -->

<script>
// ─────────────────────────────────────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────────────────────────────────────
let STATE = {
  allData: [],        // processed rows from server
  filteredData: [],
  columns: [],
  priceMin: 0,
  priceMax: 10000,
  modelTrained: false,
  modelType: '',
  currentPage: 1,
  pageSize: 50,
  locationStats: {},
  hasYearModel: false,
  uniqueVals: {},
};

const CHART_LAYOUT = {
  paper_bgcolor:'rgba(0,0,0,0)',
  plot_bgcolor:'rgba(0,0,0,0)',
  font:{color:'#e8eaf0',family:'DM Sans'},
  xaxis:{gridcolor:'rgba(255,255,255,0.05)',linecolor:'rgba(255,255,255,0.08)'},
  yaxis:{gridcolor:'rgba(255,255,255,0.05)',linecolor:'rgba(255,255,255,0.08)'},
  margin:{t:30,r:20,b:60,l:60},
  legend:{bgcolor:'rgba(0,0,0,0)'},
};
const COLORS = ['#f0a500','#e05c5c','#c084fc','#34d399','#60a5fa','#fb923c','#38bdf8','#f472b6'];

// ─────────────────────────────────────────────────────────────────────────────
// UPLOAD
// ─────────────────────────────────────────────────────────────────────────────
const uploadZone = document.getElementById('uploadZone');
const fileInput  = document.getElementById('fileInput');

uploadZone.onclick = () => fileInput.click();
uploadZone.ondragover = e => { e.preventDefault(); uploadZone.classList.add('drag-over'); };
uploadZone.ondragleave = () => uploadZone.classList.remove('drag-over');
uploadZone.ondrop = e => { e.preventDefault(); uploadZone.classList.remove('drag-over'); if(e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]); };
fileInput.onchange = () => { if(fileInput.files[0]) uploadFile(fileInput.files[0]); };

async function uploadFile(file) {
  showProgress('Uploading and parsing ' + file.name + '…', 20);
  hideAlerts();

  const form = new FormData();
  form.append('file', file);

  try {
    showProgress('Analyzing dataset…', 60);
    const res  = await fetch('/api/upload', { method:'POST', body:form });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Upload failed');

    showProgress('Rendering interface…', 90);
    STATE.allData      = data.rows;
    STATE.columns      = data.columns;
    STATE.locationStats = data.location_stats || {};
    STATE.hasYearModel  = data.has_year_model;
    STATE.uniqueVals    = data.unique_vals || {};

    initFilters();
    applyFilters();
    hideProgress();
    showAlert('success', `✅ Loaded ${data.rows.length.toLocaleString()} properties · ${data.columns.length} columns`);
    document.getElementById('mainContent').classList.remove('hidden');
    renderCharts();
    renderLocationSnapshot();
    populatePredictForm();

  } catch(err) {
    hideProgress();
    showAlert('error', '❌ ' + err.message);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// PROGRESS / ALERTS
// ─────────────────────────────────────────────────────────────────────────────
function showProgress(label, pct) {
  document.getElementById('progressWrap').style.display = 'block';
  document.getElementById('progressLabel').textContent = label;
  document.getElementById('progressFill').style.width = pct + '%';
}
function hideProgress() { document.getElementById('progressWrap').style.display = 'none'; }
function showAlert(type, msg) {
  const el = document.getElementById('alert' + type.charAt(0).toUpperCase() + type.slice(1));
  if (!el) return;
  el.textContent = msg;
  el.style.display = 'block';
  if (type !== 'error') setTimeout(() => el.style.display = 'none', 5000);
}
function hideAlerts() {
  ['alertSuccess','alertError'].forEach(id => document.getElementById(id).style.display = 'none');
}

// ─────────────────────────────────────────────────────────────────────────────
// TABS
// ─────────────────────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach((b,i) => {
    const tabs = ['overview','charts','listings','predict'];
    b.classList.toggle('active', tabs[i] === name);
  });
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  if (name === 'charts') setTimeout(renderCharts, 50);
  if (name === 'listings') renderListings();
}

// ─────────────────────────────────────────────────────────────────────────────
// FILTERS
// ─────────────────────────────────────────────────────────────────────────────
function initFilters() {
  const uv = STATE.uniqueVals;
  fillSelect('filterState', uv.state || []);
  fillSelect('filterLocation', uv.location || []);
  fillSelect('filterEnterprise', uv.enterprise || []);
  fillSelect('filterBedrooms', uv.bedrooms || []);
  fillSelect('filterFurnishing', uv.furnishing || []);

  const prices = STATE.allData.map(r => +r.price).filter(v => !isNaN(v));
  STATE.priceMin = Math.min(...prices);
  STATE.priceMax = Math.max(...prices);
  const pMin = document.getElementById('priceMin');
  const pMax = document.getElementById('priceMax');
  pMin.min = pMax.min = STATE.priceMin;
  pMin.max = pMax.max = STATE.priceMax;
  pMin.value = STATE.priceMin;
  pMax.value = STATE.priceMax;
  pMin.step = pMax.step = ((STATE.priceMax - STATE.priceMin) / 1000).toFixed(2);
  updateRangeLabel();
}

function fillSelect(id, options) {
  const el = document.getElementById(id);
  el.innerHTML = '<option value="">All</option>';
  options.forEach(o => { const opt = document.createElement('option'); opt.value = o; opt.textContent = o; el.appendChild(opt); });
}

function updateRangeLabel() {
  const mn = +document.getElementById('priceMin').value;
  const mx = +document.getElementById('priceMax').value;
  document.getElementById('priceMinLabel').textContent = '₹' + mn.toFixed(0) + 'L';
  document.getElementById('priceMaxLabel').textContent = '₹' + mx.toFixed(0) + 'L';
}

function applyFilters() {
  const state    = document.getElementById('filterState').value;
  const location = document.getElementById('filterLocation').value;
  const ent      = document.getElementById('filterEnterprise').value;
  const beds     = document.getElementById('filterBedrooms').value;
  const furn     = document.getElementById('filterFurnishing').value;
  const pMin     = +document.getElementById('priceMin').value;
  const pMax     = +document.getElementById('priceMax').value;

  STATE.filteredData = STATE.allData.filter(r => {
    if (state    && r.state      !== state)    return false;
    if (location && r.location   !== location) return false;
    if (ent      && r.enterprise !== ent)      return false;
    if (beds     && r.bedrooms   !== beds)     return false;
    if (furn     && r.furnishing !== furn)     return false;
    const p = +r.price;
    if (!isNaN(p) && (p < pMin || p > pMax))  return false;
    return true;
  });

  document.getElementById('filterCount').textContent = STATE.filteredData.length.toLocaleString();
  renderKPIs();
  renderCharts();
  STATE.currentPage = 1;
  renderListings();
}

function resetFilters() {
  ['filterState','filterLocation','filterEnterprise','filterBedrooms','filterFurnishing']
    .forEach(id => document.getElementById(id).value = '');
  document.getElementById('priceMin').value = STATE.priceMin;
  document.getElementById('priceMax').value = STATE.priceMax;
  updateRangeLabel();
  applyFilters();
}

// ─────────────────────────────────────────────────────────────────────────────
// KPI CARDS
// ─────────────────────────────────────────────────────────────────────────────
function renderKPIs() {
  const d = STATE.filteredData;
  const prices = d.map(r => +r.price).filter(v => !isNaN(v));
  const avgPrice = prices.length ? prices.reduce((a,b)=>a+b,0)/prices.length : 0;

  const ppsqft = d.map(r => +r.price_per_sqft).filter(v => !isNaN(v));
  const avgPPSqft = ppsqft.length ? ppsqft.reduce((a,b)=>a+b,0)/ppsqft.length : null;

  const sqfts = d.map(r => +r.sqft).filter(v => !isNaN(v));
  const avgSqft = sqfts.length ? sqfts.reduce((a,b)=>a+b,0)/sqfts.length : null;

  const kpis = [
    { label:'Avg Price', value: avgPrice ? '₹' + avgPrice.toFixed(2) + 'L' : '—', sub:'' },
    { label:'Avg ₹/sqft', value: avgPPSqft ? '₹' + avgPPSqft.toFixed(0) : '—', sub:'' },
    { label:'Avg Area', value: avgSqft ? avgSqft.toFixed(0) + ' sqft' : '—', sub:'' },
    { label:'Properties', value: d.length.toLocaleString(), sub:'' },
  ];

  document.getElementById('kpiGrid').innerHTML = kpis.map(k =>
    `<div class="kpi-card">
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-value">${k.value}</div>
      ${k.sub ? `<div class="kpi-sub">${k.sub}</div>` : ''}
    </div>`
  ).join('');

  // Year forecast section
  const yfSec = document.getElementById('yearForecastSection');
  yfSec.classList.toggle('hidden', !STATE.hasYearModel);
}

// ─────────────────────────────────────────────────────────────────────────────
// YEAR FORECAST
// ─────────────────────────────────────────────────────────────────────────────
async function runYearForecast() {
  const year = +document.getElementById('forecastYear').value;
  try {
    const res  = await fetch('/api/year_forecast', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({year}) });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    document.getElementById('forecastYearLabel').textContent = year;
    document.getElementById('forecastVal').textContent = '₹' + data.price.toFixed(2) + 'L';
    document.getElementById('forecastResult').classList.remove('hidden');
  } catch(e) { showAlert('error', e.message); }
}

// ─────────────────────────────────────────────────────────────────────────────
// LOCATION SNAPSHOT TABLE
// ─────────────────────────────────────────────────────────────────────────────
function renderLocationSnapshot() {
  const ls = STATE.locationStats;
  const key = ls.location ? 'location' : (ls.state ? 'state' : null);
  if (!key) return;

  const sec = document.getElementById('locationSnapshotSection');
  sec.classList.remove('hidden');

  const rows = ls[key];
  const cols = Object.keys(rows[0]).filter(c => c !== key);
  const tbl = `<table>
    <thead><tr>
      <th>${key.toUpperCase()}</th>
      ${cols.map(c=>`<th>${c.replace(/_/g,' ').toUpperCase()}</th>`).join('')}
    </tr></thead>
    <tbody>
    ${rows.map(r=>`<tr>
      <td><b>${r[key]}</b></td>
      ${cols.map(c=>`<td>${typeof r[c]==='number'?`<span class="loc-stat-chip">₹${(+r[c]).toLocaleString()}</span>`:r[c]}</td>`).join('')}
    </tr>`).join('')}
    </tbody>
  </table>`;
  document.getElementById('locationTable').innerHTML = tbl;
}

// ─────────────────────────────────────────────────────────────────────────────
// CHARTS
// ─────────────────────────────────────────────────────────────────────────────
function renderCharts() {
  const d = STATE.filteredData;
  if (!d.length) return;

  const cfg = { responsive:true, displayModeBar:false };

  // Histogram
  const prices = d.map(r => +r.price).filter(v => !isNaN(v));
  Plotly.react('chartHist', [{
    x: prices, type:'histogram', nbinsx:40,
    marker:{ color:'#f0a500', line:{width:0}, opacity:0.85 },
    name:'Properties'
  }], { ...CHART_LAYOUT, xaxis:{...CHART_LAYOUT.xaxis, title:'Price (Lakhs ₹)'},
       yaxis:{...CHART_LAYOUT.yaxis, title:'Count'} }, cfg);

  // Year-wise
  const hasYear = d.some(r => r.year);
  document.getElementById('chartYearWrap').style.display = hasYear ? '' : 'none';
  if (hasYear) {
    const yrMap = {};
    d.forEach(r => { if(r.year && r.price){ const y=Math.round(+r.year); if(!yrMap[y]){yrMap[y]=[];} yrMap[y].push(+r.price); }});
    const yrs = Object.keys(yrMap).sort();
    const avgs = yrs.map(y => yrMap[y].reduce((a,b)=>a+b,0)/yrMap[y].length);
    Plotly.react('chartYear', [{
      x:yrs, y:avgs, type:'bar',
      marker:{ color:avgs, colorscale:[[0,'#1a2236'],[1,'#f0a500']], showscale:false },
      name:'Avg Price'
    }], { ...CHART_LAYOUT, xaxis:{...CHART_LAYOUT.xaxis,title:'Year Built'},
          yaxis:{...CHART_LAYOUT.yaxis,title:'Avg Price (₹L)'} }, cfg);
  }

  // Scatter
  const hasSqft = d.some(r => r.sqft);
  document.getElementById('chartScatterWrap').style.display = hasSqft ? '' : 'none';
  if (hasSqft) {
    const byBed = {};
    d.forEach(r => {
      const key = r.bedrooms || 'Unknown';
      if(!byBed[key]) byBed[key] = {x:[],y:[]};
      if(r.sqft && r.price){ byBed[key].x.push(+r.sqft); byBed[key].y.push(+r.price); }
    });
    const traces = Object.entries(byBed).map(([bed, pts], i) => ({
      x:pts.x, y:pts.y, mode:'markers', name:bed, type:'scatter',
      marker:{ color:COLORS[i%COLORS.length], size:7, opacity:0.75 }
    }));
    Plotly.react('chartScatter', traces, { ...CHART_LAYOUT,
      xaxis:{...CHART_LAYOUT.xaxis,title:'Area (sqft)'},
      yaxis:{...CHART_LAYOUT.yaxis,title:'Price (₹L)'} }, cfg);
  }

  // Location bar
  const locCol = STATE.columns.includes('location') ? 'location' : (STATE.columns.includes('state') ? 'state' : null);
  document.getElementById('chartLocWrap').style.display = locCol ? '' : 'none';
  if (locCol) {
    const locMap = {};
    d.forEach(r => { if(r[locCol]&&r.price){ if(!locMap[r[locCol]]) locMap[r[locCol]]=[]; locMap[r[locCol]].push(+r.price); }});
    const entries = Object.entries(locMap).map(([l,ps])=>[l, ps.reduce((a,b)=>a+b,0)/ps.length])
      .sort((a,b)=>b[1]-a[1]).slice(0,20);
    Plotly.react('chartLoc', [{
      x: entries.map(e=>e[1]), y:entries.map(e=>e[0]),
      type:'bar', orientation:'h',
      marker:{ color:entries.map(e=>e[1]), colorscale:[[0,'#1a2236'],[0.5,'#e05c5c'],[1,'#f0a500']], showscale:false }
    }], { ...CHART_LAYOUT, height: Math.max(300, entries.length*34),
      xaxis:{...CHART_LAYOUT.xaxis,title:'Avg Price (₹L)'},
      yaxis:{...CHART_LAYOUT.yaxis},
      margin:{...CHART_LAYOUT.margin, l:140} }, cfg);
  }

  // Furnishing box
  const hasFurn = d.some(r => r.furnishing);
  document.getElementById('chartFurnWrap').style.display = hasFurn ? '' : 'none';
  if (hasFurn) {
    const furnMap = {};
    d.forEach(r => { if(r.furnishing&&r.price){ if(!furnMap[r.furnishing]) furnMap[r.furnishing]=[]; furnMap[r.furnishing].push(+r.price); }});
    const traces = Object.entries(furnMap).map(([f, ps], i) => ({
      y:ps, type:'box', name:f, marker:{color:COLORS[i%COLORS.length]}, boxpoints:'outliers'
    }));
    Plotly.react('chartFurn', traces, { ...CHART_LAYOUT,
      yaxis:{...CHART_LAYOUT.yaxis, title:'Price (₹L)'} }, cfg);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// LISTINGS TABLE
// ─────────────────────────────────────────────────────────────────────────────
const PREFERRED = ['location','enterprise','state','price','sqft','bedrooms','bathrooms','furnishing','floor','total_floors','parking','facing','metro_distance_km','price_per_sqft'];

function renderListings() {
  const search = document.getElementById('listingSearch').value.toLowerCase();
  let rows = STATE.filteredData;
  if (search) rows = rows.filter(r => JSON.stringify(r).toLowerCase().includes(search));

  const cols = PREFERRED.filter(c => STATE.columns.includes(c));
  if (!cols.length) return;

  const total = rows.length;
  const pages = Math.ceil(total / STATE.pageSize);
  if (STATE.currentPage > pages) STATE.currentPage = 1;
  const start = (STATE.currentPage - 1) * STATE.pageSize;
  const pageRows = rows.slice(start, start + STATE.pageSize);

  const furnBadge = f => {
    if (!f) return '';
    const cls = f.toLowerCase().includes('unfurn') ? 'unfurnished' : f.toLowerCase().includes('semi') ? 'semi' : 'furnished';
    return `<span class="badge badge-${cls}">${f}</span>`;
  };

  const thead = `<thead><tr>${cols.map(c=>`<th>${c.replace(/_/g,' ').toUpperCase()}</th>`).join('')}</tr></thead>`;
  const tbody = '<tbody>' + pageRows.map(r =>
    `<tr>${cols.map(c => {
      const v = r[c];
      if (c === 'price') return `<td class="price-cell">₹${(+v).toFixed(2)}L</td>`;
      if (c === 'furnishing') return `<td>${furnBadge(v)}</td>`;
      if (c === 'price_per_sqft' && v) return `<td>₹${(+v).toFixed(0)}</td>`;
      if (c === 'metro_distance_km' && v) return `<td>${(+v).toFixed(1)} km</td>`;
      return `<td>${v ?? '—'}</td>`;
    }).join('')}</tr>`
  ).join('') + '</tbody>';

  document.getElementById('listingsTable').innerHTML = `<table>${thead}${tbody}</table>`;

  // Pagination
  const bar = document.getElementById('paginationBar');
  if (pages <= 1) { bar.innerHTML = ''; return; }
  let pHtml = `<span class="page-info">${start+1}–${Math.min(start+STATE.pageSize,total)} of ${total.toLocaleString()}</span>`;
  pHtml += `<button class="page-btn" onclick="goPage(${STATE.currentPage-1})" ${STATE.currentPage===1?'disabled':''}>‹</button>`;
  const range = [...new Set([1,2,STATE.currentPage-1,STATE.currentPage,STATE.currentPage+1,pages-1,pages])].filter(p=>p>=1&&p<=pages).sort((a,b)=>a-b);
  let prev = 0;
  range.forEach(p => {
    if (prev && p - prev > 1) pHtml += `<span style="color:var(--muted)">…</span>`;
    pHtml += `<button class="page-btn ${p===STATE.currentPage?'active':''}" onclick="goPage(${p})">${p}</button>`;
    prev = p;
  });
  pHtml += `<button class="page-btn" onclick="goPage(${STATE.currentPage+1})" ${STATE.currentPage===pages?'disabled':''}>›</button>`;
  bar.innerHTML = pHtml;
}

function goPage(p) { STATE.currentPage = p; renderListings(); }

function downloadCSV() {
  const cols = PREFERRED.filter(c => STATE.columns.includes(c));
  const search = document.getElementById('listingSearch').value.toLowerCase();
  let rows = STATE.filteredData;
  if (search) rows = rows.filter(r => JSON.stringify(r).toLowerCase().includes(search));

  const header = cols.join(',');
  const body = rows.map(r => cols.map(c => {
    const v = r[c] ?? '';
    return typeof v === 'string' && v.includes(',') ? `"${v}"` : v;
  }).join(',')).join('\n');
  const blob = new Blob([header + '\n' + body], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'filtered_properties.csv';
  a.click();
  return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// POPULATE PREDICT FORM
// ─────────────────────────────────────────────────────────────────────────────
function populatePredictForm() {
  const uv = STATE.uniqueVals;
  const toggle = (wrapId, selId, opts) => {
    const wrap = document.getElementById(wrapId);
    const sel  = document.getElementById(selId);
    if (opts && opts.length) {
      wrap.classList.remove('hidden');
      sel.innerHTML = opts.map(o=>`<option>${o}</option>`).join('');
    } else {
      wrap.classList.add('hidden');
    }
  };
  toggle('fFurnishWrap', 'fFurnishing', uv.furnishing || []);
  toggle('fFacingWrap',  'fFacing',    uv.facing     || []);
  toggle('fLocationWrap','fLocation',  uv.location   || []);
}

// ─────────────────────────────────────────────────────────────────────────────
// TRAIN MODEL
// ─────────────────────────────────────────────────────────────────────────────
async function trainModel() {
  const btn = document.getElementById('trainBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Training…';
  const modelType = document.getElementById('modelType').value;

  try {
    const res  = await fetch('/api/train', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ model_type: modelType }) });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    document.getElementById('r2Score').textContent = data.r2.toFixed(3);
    document.getElementById('maeScore').textContent = '₹' + data.mae.toFixed(2) + 'L';
    document.getElementById('nTrain').textContent = data.n_train.toLocaleString();
    document.getElementById('trainResult').classList.remove('hidden');
    document.getElementById('estimateForm').classList.remove('hidden');
    STATE.modelTrained = true;
    STATE.modelType = modelType;
    showAlert('success', '✅ Model trained! R² = ' + data.r2.toFixed(3));
  } catch(e) {
    showAlert('error', '❌ ' + e.message);
  }
  btn.disabled = false;
  btn.textContent = '🚀 Train Model';
}

// ─────────────────────────────────────────────────────────────────────────────
// PREDICT
// ─────────────────────────────────────────────────────────────────────────────
async function estimatePrice() {
  const bedroomMap = {'1 RK':0,'1 BHK':1,'2 BHK':2,'3 BHK':3,'4 BHK':4,'5 BHK':5};
  const bedStr = document.getElementById('fBedrooms').value;

  const input = {
    sqft:        +document.getElementById('fSqft').value,
    bedrooms:    bedroomMap[bedStr] ?? 1,
    bathrooms:   +document.getElementById('fBathrooms').value,
    floor:       +document.getElementById('fFloor').value,
    total_floors:+document.getElementById('fTotalFloors').value,
    parking:     +document.getElementById('fParking').value,
    age_years:   +document.getElementById('fAge').value,
    metro_distance_km: +document.getElementById('fMetro').value,
  };
  if (!document.getElementById('fFurnishWrap').classList.contains('hidden'))
    input.furnishing = document.getElementById('fFurnishing').value;
  if (!document.getElementById('fFacingWrap').classList.contains('hidden'))
    input.facing = document.getElementById('fFacing').value;
  if (!document.getElementById('fLocationWrap').classList.contains('hidden'))
    input.location = document.getElementById('fLocation').value;

  try {
    const res  = await fetch('/api/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(input) });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    document.getElementById('predPrice').textContent = '₹' + data.price.toFixed(2) + 'L';
    document.getElementById('predAlgo').textContent = 'via ' + STATE.modelType;
    document.getElementById('predResult').classList.remove('hidden');
  } catch(e) { showAlert('error', '❌ ' + e.message); }
}

// Handle resize
window.addEventListener('resize', () => {
  if (STATE.filteredData.length) renderCharts();
});
</script>
</body>
</html>"""

# ─────────────────────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/upload", methods=["POST"])
def api_upload():
    try:
        f = request.files.get("file")
        if not f:
            return jsonify({"error": "No file provided"}), 400

        file_bytes = f.read()
        df_raw = load_file(file_bytes, f.filename)
        df = analyzer.process_dataframe(df_raw)

        # Build unique values for UI
        unique_vals = {}
        for col in ["state", "location", "enterprise", "bedrooms", "furnishing", "facing"]:
            if col in df.columns:
                unique_vals[col] = sorted(df[col].dropna().unique().tolist(), key=str)

        # Location stats — convert to JSON-serializable format
        loc_stats = {}
        for k, v in analyzer.location_stats.items():
            loc_stats[k] = v.reset_index().to_dict(orient="records")

        # Serialize rows — replace NaN with None
        rows = json.loads(df.to_json(orient="records"))

        return jsonify({
            "rows": rows,
            "columns": df.columns.tolist(),
            "has_year_model": analyzer.year_model is not None,
            "location_stats": loc_stats,
            "unique_vals": unique_vals,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/year_forecast", methods=["POST"])
def api_year_forecast():
    try:
        year = request.json.get("year", 2027)
        price = analyzer.predict_year_price(int(year))
        return jsonify({"price": price})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/train", methods=["POST"])
def api_train():
    try:
        model_type = request.json.get("model_type", "Random Forest")
        results = analyzer.train_model(model_type=model_type)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        input_dict = request.json
        price = analyzer.predict(input_dict)
        return jsonify({"price": price})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*55)
    print("  🏙️  PropertyIQ — AI Property Price Intelligence")
    print("="*55)
    print(f"  ➜  Local:   http://localhost:{port}")
    print(f"  ➜  Press Ctrl+C to stop")
    print("="*55 + "\n")

    def open_browser():
        import time; time.sleep(1.2)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=port, debug=False)

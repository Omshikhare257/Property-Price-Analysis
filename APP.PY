import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import io
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# FILE PARSERS — support CSV, Excel (.xls/.xlsx), PDF, PPTX, JPG/PNG
# ─────────────────────────────────────────────────────────────────────────────

def parse_csv(file_bytes, filename):
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
    # Return the largest sheet
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
        combined = combined.replace("", np.nan).dropna(how="all")
        return combined
    raise ValueError("No tables found in PDF. Try uploading a CSV/Excel instead.")


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
        # Try comma/tab separation
        sep = "," if "," in lines[0] else "\t"
        data = [l.split(sep) for l in lines]
        df = pd.DataFrame(data[1:], columns=data[0])
        return df.replace("", np.nan).dropna(how="all")
    except ImportError:
        raise ValueError("pytesseract not available for image parsing.")


def load_file(uploaded_file):
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()
    if filename.endswith(".csv"):
        return parse_csv(file_bytes, filename)
    elif filename.endswith((".xlsx", ".xls", ".xlsm")):
        return parse_excel(file_bytes, filename)
    elif filename.endswith(".pdf"):
        return parse_pdf(file_bytes)
    elif filename.endswith(".pptx"):
        return parse_pptx(file_bytes)
    elif filename.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return parse_image(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: '{filename}'. Supported: CSV, Excel, PDF, PPTX, JPG/PNG.")


# ─────────────────────────────────────────────────────────────────────────────
# CORE ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

BEDROOM_TO_NUM = {"1 RK": 0, "1 BHK": 1, "2 BHK": 2, "3 BHK": 3, "4 BHK": 4, "5 BHK": 5}
NUM_TO_BEDROOM = {v: k for k, v in BEDROOM_TO_NUM.items()}


def bedroom_to_str(x):
    if pd.isna(x):
        return np.nan
    try:
        xi = int(float(x))
        return NUM_TO_BEDROOM.get(xi, f"{xi} BHK")
    except (ValueError, TypeError):
        return x


def bedroom_to_num(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        if x in BEDROOM_TO_NUM:
            return BEDROOM_TO_NUM[x]
        parts = x.split()
        if parts and parts[0].isdigit():
            return int(parts[0])
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return np.nan


class PropertyAnalyzer:
    def __init__(self):
        self.model = None
        self.model_type = None
        self.scaler = StandardScaler()
        self.features = []
        self.year_model = None
        self.location_stats = {}

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Normalize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        numeric_cols = [
            "price", "sqft", "bedrooms", "bathrooms", "floor",
            "total_floors", "age_years", "parking",
            "metro_distance_km", "price_per_sqft"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Bedrooms → display string
        if "bedrooms" in df.columns:
            df["bedrooms"] = df["bedrooms"].apply(bedroom_to_str)

        # Derive year from age
        if "age_years" in df.columns and "year" not in df.columns:
            current = pd.Timestamp.now().year
            df["year"] = (current - df["age_years"]).where(df["age_years"].notna())

        df = df.dropna(subset=["price"])

        # Compute location stats
        for grp_col in ["location", "state"]:
            if grp_col in df.columns:
                agg = df.groupby(grp_col).agg(
                    mean_price=("price", "mean"),
                    median_price=("price", "median"),
                    min_price=("price", "min"),
                    max_price=("price", "max"),
                    count=("price", "count"),
                ).round(2)
                if "price_per_sqft" in df.columns:
                    agg["avg_ppsqft"] = df.groupby(grp_col)["price_per_sqft"].mean().round(2)
                self.location_stats[grp_col] = agg

        # Train year model
        if "year" in df.columns:
            valid = df[["year", "price"]].dropna()
            if len(valid) > 5:
                self.year_model = LinearRegression()
                self.year_model.fit(valid[["year"]], valid["price"])

        return df

    def predict_year_price(self, year: int) -> float:
        if self.year_model is None:
            raise ValueError("Year model not trained.")
        return float(self.year_model.predict([[year]])[0])

    def train_model(self, df: pd.DataFrame, model_type: str = "Random Forest") -> dict:
        df_m = df.copy()
        # Convert bedrooms to numeric for training
        if "bedrooms" in df_m.columns:
            df_m["bedrooms"] = df_m["bedrooms"].apply(bedroom_to_num)

        numeric_features = [
            c for c in ["sqft", "bedrooms", "bathrooms", "floor",
                         "total_floors", "age_years", "parking", "metro_distance_km"]
            if c in df_m.columns
        ]
        cat_features = [
            c for c in ["state", "location", "furnishing", "facing"]
            if c in df_m.columns
        ]

        if not numeric_features:
            raise ValueError("No numeric features found to train model.")

        df_m = df_m[numeric_features + cat_features + ["price"]].dropna(subset=["price"])
        if len(df_m) < 10:
            raise ValueError(f"Only {len(df_m)} rows after cleaning — need at least 10.")

        X = df_m[numeric_features].copy()
        if cat_features:
            dummies = pd.get_dummies(df_m[cat_features], drop_first=True)
            X = pd.concat([X, dummies], axis=1)

        y = df_m["price"]
        self.features = X.columns.tolist()
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        self.model = models[model_type]
        self.model.fit(X_train, y_train)
        self.model_type = model_type

        y_pred = self.model.predict(X_test)
        return {
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    def predict(self, input_dict: dict) -> float:
        if self.model is None:
            raise ValueError("Model not trained yet.")
        row = {}
        for f in self.features:
            row[f] = input_dict.get(f, 0)
        df_in = pd.DataFrame([row])
        # Align columns
        for col in self.features:
            if col not in df_in.columns:
                df_in[col] = 0
        df_in = df_in[self.features]
        scaled = self.scaler.transform(df_in)
        return float(self.model.predict(scaled)[0])


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────

PAGE_CONFIG_SET = False


def apply_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    :root {
        --bg: #0b0f1a;
        --surface: #131929;
        --surface2: #1a2236;
        --accent: #f0a500;
        --accent2: #e05c5c;
        --text: #e8eaf0;
        --muted: #7a859a;
        --border: rgba(255,255,255,0.07);
        --radius: 14px;
    }

    html, body, [class*="css"], .stApp {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Hero title */
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f0a500 0%, #e05c5c 60%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.05rem;
        color: var(--muted);
        margin-bottom: 2rem;
    }

    /* Cards */
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: var(--accent);
    }
    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--muted);
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.7rem;
        font-weight: 700;
        color: var(--accent);
    }

    /* Section headers */
    .section-header {
        font-family: 'Syne', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text);
        border-left: 4px solid var(--accent);
        padding-left: 0.75rem;
        margin: 1.8rem 0 1rem 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: var(--muted) !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: var(--surface2) !important;
        border: 2px dashed rgba(240, 165, 0, 0.35) !important;
        border-radius: var(--radius) !important;
        padding: 1rem !important;
    }

    /* Selectboxes & inputs */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #f0a500, #e05c5c) !important;
        color: #0b0f1a !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        transition: opacity 0.2s, transform 0.15s !important;
        letter-spacing: 0.03em;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
        transform: translateY(-1px) !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: var(--radius) !important;
        overflow: hidden;
        border: 1px solid var(--border) !important;
    }

    /* Alerts */
    .stAlert {
        border-radius: var(--radius) !important;
        border: 1px solid var(--border) !important;
    }

    /* Tab bar */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface) !important;
        border-radius: var(--radius) !important;
        gap: 4px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        color: var(--muted) !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f0a500, #e05c5c) !important;
        color: #0b0f1a !important;
    }

    /* Plotly chart backgrounds */
    .js-plotly-plot .plotly .bg {
        fill: transparent !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--accent) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--surface2); border-radius: 3px; }

    /* Metrics native Streamlit */
    [data-testid="metric-container"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1rem !important;
    }
    [data-testid="metric-container"] label {
        color: var(--muted) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--accent) !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    </style>
    """, unsafe_allow_html=True)


CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e8eaf0",
    font_family="DM Sans",
    colorway=["#f0a500", "#e05c5c", "#c084fc", "#34d399", "#60a5fa", "#fb923c"],
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.1)"),
)


def styled_chart(fig):
    fig.update_layout(**CHART_THEME)
    return fig


def safe_format(val, fmt="₹{:,.2f}L"):
    try:
        return fmt.format(val)
    except Exception:
        return str(val)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_analyzer() -> PropertyAnalyzer:
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = PropertyAnalyzer()
    return st.session_state.analyzer


def get_df() -> pd.DataFrame | None:
    return st.session_state.get("df", None)


def get_model_results() -> dict | None:
    return st.session_state.get("model_results", None)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="PropertyIQ — Price Intelligence",
        page_icon="🏙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_css()

    analyzer = get_analyzer()

    # ── HERO ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="hero-title">🏙️ PropertyIQ</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">AI-powered property price intelligence · '
        'Upload CSV · Excel · PDF · PPTX · Image</div>',
        unsafe_allow_html=True,
    )

    # ── FILE UPLOAD ───────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload your property dataset",
        type=["csv", "xlsx", "xls", "xlsm", "pdf", "pptx", "jpg", "jpeg", "png"],
        help="Supports CSV, Excel (xls/xlsx), PDF tables, PPTX tables, and image files with tabular data.",
    )

    if uploaded is not None:
        file_key = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("_last_file") != file_key:
            # New file — reset state
            st.session_state.analyzer = PropertyAnalyzer()
            st.session_state.model_results = None
            st.session_state._last_file = file_key
            analyzer = st.session_state.analyzer

            with st.spinner(f"📂 Parsing **{uploaded.name}**…"):
                try:
                    raw_df = load_file(uploaded)
                    df = analyzer.process_dataframe(raw_df)
                    st.session_state.df = df
                    st.success(
                        f"✅ Loaded **{len(df):,}** properties · "
                        f"**{len(df.columns)}** columns detected."
                    )
                except Exception as e:
                    st.error(f"❌ Failed to load file: {e}")
                    st.stop()

    df = get_df()

    if df is None:
        # ── LANDING / INSTRUCTIONS ─────────────────────────────────────────
        col_l, col_r = st.columns([1, 1], gap="large")
        with col_l:
            st.markdown('<div class="section-header">Supported formats</div>', unsafe_allow_html=True)
            formats = [
                ("📄 CSV", "Standard comma-separated values"),
                ("📊 Excel", ".xlsx / .xls / .xlsm files"),
                ("📑 PDF", "Extracts tables from PDF pages"),
                ("📽️ PPTX", "Reads tables embedded in slides"),
                ("🖼️ JPG / PNG", "OCR-based table extraction"),
            ]
            for icon_name, desc in formats:
                st.markdown(
                    f"<div style='padding:0.6rem 1rem;margin:0.35rem 0;"
                    f"background:#131929;border-radius:10px;border:1px solid rgba(255,255,255,0.06)'>"
                    f"<b style='color:#f0a500'>{icon_name}</b> &nbsp; "
                    f"<span style='color:#7a859a;font-size:0.9rem'>{desc}</span></div>",
                    unsafe_allow_html=True,
                )
        with col_r:
            st.markdown('<div class="section-header">Expected columns</div>', unsafe_allow_html=True)
            st.code(
                "location, state, price, sqft, bedrooms, bathrooms,\n"
                "floor, total_floors, age_years, parking,\n"
                "metro_distance_km, furnishing, facing,\n"
                "enterprise, price_per_sqft",
                language="text",
            )
            st.markdown(
                "<span style='color:#7a859a;font-size:0.85rem'>"
                "All columns are optional — the app adapts to whatever is present.</span>",
                unsafe_allow_html=True,
            )
        return

    # ─────────────────────────────────────────────────────────────────────────
    # SIDEBAR FILTERS
    # ─────────────────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        "<h2 style='font-family:Syne,sans-serif;color:#f0a500;margin-bottom:0.5rem'>"
        "🎛️ Filters</h2>",
        unsafe_allow_html=True,
    )

    df_filtered = df.copy()

    # State → Location cascade
    if "state" in df.columns:
        states = sorted(df["state"].dropna().unique().tolist())
        sel_state = st.sidebar.selectbox("State", ["All"] + states)
        if sel_state != "All":
            df_filtered = df_filtered[df_filtered["state"] == sel_state]

    if "location" in df.columns:
        locs = sorted(df_filtered["location"].dropna().unique().tolist())
        sel_location = st.sidebar.selectbox("Location", ["All"] + locs)
        if sel_location != "All":
            df_filtered = df_filtered[df_filtered["location"] == sel_location]

    # Enterprise
    if "enterprise" in df.columns:
        ents = sorted(df["enterprise"].dropna().unique().tolist())
        sel_ent = st.sidebar.selectbox("Enterprise / Builder", ["All"] + ents)
        if sel_ent != "All":
            df_filtered = df_filtered[df_filtered["enterprise"] == sel_ent]

    # Price range
    p_min = float(df["price"].min())
    p_max = float(df["price"].max())
    if p_min < p_max:
        price_range = st.sidebar.slider(
            "Price Range (Lakhs ₹)",
            min_value=p_min,
            max_value=p_max,
            value=(p_min, p_max),
            format="₹%.1f",
        )
        df_filtered = df_filtered[
            (df_filtered["price"] >= price_range[0])
            & (df_filtered["price"] <= price_range[1])
        ]

    # Bedrooms
    if "bedrooms" in df.columns:
        bed_opts = sorted(
            df["bedrooms"].dropna().unique().tolist(),
            key=lambda x: bedroom_to_num(x) if bedroom_to_num(x) is not None else 999,
        )
        sel_bed = st.sidebar.selectbox("Bedrooms", ["All"] + bed_opts)
        if sel_bed != "All":
            df_filtered = df_filtered[df_filtered["bedrooms"] == sel_bed]

    # Furnishing
    if "furnishing" in df.columns:
        furn_opts = sorted(df["furnishing"].dropna().unique().tolist())
        sel_furn = st.sidebar.multiselect("Furnishing", furn_opts)
        if sel_furn:
            df_filtered = df_filtered[df_filtered["furnishing"].isin(sel_furn)]

    st.sidebar.markdown(
        f"<div style='margin-top:1rem;padding:0.75rem;background:#1a2236;"
        f"border-radius:10px;text-align:center'>"
        f"<span style='color:#7a859a;font-size:0.8rem'>SHOWING</span><br>"
        f"<span style='font-family:Syne,sans-serif;font-size:1.5rem;color:#f0a500;font-weight:700'>"
        f"{len(df_filtered):,}</span> "
        f"<span style='color:#7a859a;font-size:0.85rem'>/ {len(df):,} properties</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if len(df_filtered) == 0:
        st.warning("⚠️ No properties match current filters. Adjust the sidebar.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # TABS
    # ─────────────────────────────────────────────────────────────────────────
    tab_overview, tab_charts, tab_listings, tab_predict = st.tabs(
        ["📊 Overview", "📈 Charts", "🏠 Listings", "🎯 Price Prediction"]
    )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ════════════════════════════════════════════════════════════════════════
    with tab_overview:
        # KPI Row
        kpi_cols = st.columns(4)
        kpis = [
            ("Avg Price", safe_format(df_filtered["price"].mean())),
            (
                "Avg ₹/sqft",
                safe_format(df_filtered["price_per_sqft"].mean(), "₹{:,.0f}")
                if "price_per_sqft" in df_filtered.columns
                else "N/A",
            ),
            (
                "Avg Area",
                safe_format(df_filtered["sqft"].mean(), "{:,.0f} sqft")
                if "sqft" in df_filtered.columns
                else "N/A",
            ),
            ("Properties", f"{len(df_filtered):,}"),
        ]
        for col, (label, val) in zip(kpi_cols, kpis):
            col.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Year price prediction
        if analyzer.year_model is not None:
            st.markdown('<div class="section-header">Year Price Forecast</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                pred_year = st.number_input(
                    "Forecast Year", min_value=1990, max_value=2060, value=2027
                )
            with c2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("📈 Forecast"):
                    try:
                        pred_val = analyzer.predict_year_price(pred_year)
                        st.session_state["year_pred"] = (pred_year, pred_val)
                    except Exception as e:
                        st.error(str(e))
            if "year_pred" in st.session_state:
                yr, pv = st.session_state["year_pred"]
                with c3:
                    st.markdown(
                        f"<div style='background:#1a2236;border-radius:12px;padding:1rem;margin-top:0.5rem'>"
                        f"<span style='color:#7a859a;font-size:0.8rem'>PREDICTED AVG PRICE FOR {yr}</span><br>"
                        f"<span style='font-family:Syne,sans-serif;font-size:2rem;color:#f0a500;font-weight:800'>"
                        f"₹{pv:,.2f}L</span></div>",
                        unsafe_allow_html=True,
                    )

        # Location summary table
        if analyzer.location_stats:
            st.markdown('<div class="section-header">Location Snapshot</div>', unsafe_allow_html=True)
            grp_key = "location" if "location" in analyzer.location_stats else "state"
            loc_df = analyzer.location_stats[grp_key].reset_index()
            loc_df.columns = [
                c.replace("_", " ").title() for c in loc_df.columns
            ]
            st.dataframe(loc_df, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — CHARTS
    # ════════════════════════════════════════════════════════════════════════
    with tab_charts:
        # Price distribution
        st.markdown('<div class="section-header">Price Distribution</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            df_filtered,
            x="price",
            nbins=40,
            labels={"price": "Price (Lakhs ₹)", "count": "Properties"},
            color_discrete_sequence=["#f0a500"],
        )
        fig_hist.update_traces(marker_line_width=0, opacity=0.85)
        st.plotly_chart(styled_chart(fig_hist), use_container_width=True)

        # Year-wise analysis
        if "year" in df_filtered.columns:
            st.markdown('<div class="section-header">Year-wise Avg Price</div>', unsafe_allow_html=True)
            yr_df = (
                df_filtered.groupby("year")["price"]
                .mean()
                .reset_index()
                .rename(columns={"price": "avg_price"})
            )
            fig_yr = px.bar(
                yr_df,
                x="year",
                y="avg_price",
                labels={"year": "Year Built", "avg_price": "Avg Price (₹L)"},
                color="avg_price",
                color_continuous_scale=["#1a2236", "#f0a500"],
            )
            fig_yr.update_coloraxes(showscale=False)
            st.plotly_chart(styled_chart(fig_yr), use_container_width=True)

        # Price vs Area scatter
        if "sqft" in df_filtered.columns:
            st.markdown('<div class="section-header">Price vs Area</div>', unsafe_allow_html=True)
            scatter_color = "bedrooms" if "bedrooms" in df_filtered.columns else None
            hover = [c for c in ["location", "furnishing", "enterprise"] if c in df_filtered.columns]
            fig_sc = px.scatter(
                df_filtered,
                x="sqft",
                y="price",
                color=scatter_color,
                hover_data=hover,
                labels={"sqft": "Area (sqft)", "price": "Price (₹L)"},
                opacity=0.75,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_sc.update_traces(marker_size=7)
            st.plotly_chart(styled_chart(fig_sc), use_container_width=True)

        # Location price comparison
        loc_col = "location" if "location" in df_filtered.columns else (
            "state" if "state" in df_filtered.columns else None
        )
        if loc_col:
            st.markdown('<div class="section-header">Avg Price by Location</div>', unsafe_allow_html=True)
            loc_avg = (
                df_filtered.groupby(loc_col)["price"]
                .mean()
                .sort_values(ascending=False)
                .head(20)
                .reset_index()
            )
            fig_bar = px.bar(
                loc_avg,
                x="price",
                y=loc_col,
                orientation="h",
                labels={"price": "Avg Price (₹L)", loc_col: ""},
                color="price",
                color_continuous_scale=["#1a2236", "#e05c5c", "#f0a500"],
            )
            fig_bar.update_coloraxes(showscale=False)
            fig_bar.update_layout(height=max(300, len(loc_avg) * 32))
            st.plotly_chart(styled_chart(fig_bar), use_container_width=True)

        # Furnishing breakdown
        if "furnishing" in df_filtered.columns:
            st.markdown('<div class="section-header">Furnishing vs Price</div>', unsafe_allow_html=True)
            fig_box = px.box(
                df_filtered,
                x="furnishing",
                y="price",
                color="furnishing",
                labels={"price": "Price (₹L)", "furnishing": ""},
                color_discrete_sequence=["#f0a500", "#e05c5c", "#c084fc"],
            )
            st.plotly_chart(styled_chart(fig_box), use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — LISTINGS
    # ════════════════════════════════════════════════════════════════════════
    with tab_listings:
        st.markdown('<div class="section-header">Property Listings</div>', unsafe_allow_html=True)

        preferred_cols = [
            "location", "enterprise", "state", "price", "sqft", "bedrooms",
            "bathrooms", "furnishing", "floor", "total_floors", "parking",
            "facing", "metro_distance_km", "price_per_sqft",
        ]
        disp_cols = [c for c in preferred_cols if c in df_filtered.columns]
        if not disp_cols:
            disp_cols = df_filtered.columns.tolist()

        fmt = {"price": "₹{:,.2f}L"}
        if "sqft" in disp_cols:
            fmt["sqft"] = "{:,.0f}"
        if "price_per_sqft" in disp_cols:
            fmt["price_per_sqft"] = "₹{:,.0f}"
        if "metro_distance_km" in disp_cols:
            fmt["metro_distance_km"] = "{:.1f} km"

        show_df = df_filtered[disp_cols].sort_values("price", ascending=False)

        # Search box
        search = st.text_input("🔍 Search in listings", placeholder="e.g. Whitefield, 3 BHK…")
        if search:
            mask = show_df.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False)
            ).any(axis=1)
            show_df = show_df[mask]
            st.caption(f'{len(show_df):,} results for "{search}"')

        st.dataframe(show_df.style.format(fmt), use_container_width=True, hide_index=True)

        # Download button
        csv_bytes = show_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download filtered listings (CSV)",
            data=csv_bytes,
            file_name="filtered_properties.csv",
            mime="text/csv",
        )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — PRICE PREDICTION
    # ════════════════════════════════════════════════════════════════════════
    with tab_predict:
        st.markdown('<div class="section-header">Train Prediction Model</div>', unsafe_allow_html=True)

        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            model_type = st.selectbox(
                "Algorithm",
                ["Random Forest", "Gradient Boosting", "Linear Regression"],
                help="Random Forest & Gradient Boosting generally outperform Linear Regression on property data.",
            )
        with col_m2:
            st.markdown("<br>", unsafe_allow_html=True)
            train_btn = st.button("🚀 Train Model")

        if train_btn:
            with st.spinner("Training… this may take a moment."):
                try:
                    results = analyzer.train_model(df, model_type=model_type)
                    st.session_state.model_results = results
                    st.success("Model trained successfully!")
                except Exception as e:
                    st.error(f"Training failed: {e}")

        results = get_model_results()
        if results:
            r2, mae, n_train, n_test = (
                results["r2"], results["mae"], results["n_train"], results["n_test"]
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("R² Score", f"{r2:.3f}", help="1.0 = perfect; >0.8 = good")
            m2.metric("Mean Abs Error", safe_format(mae))
            m3.metric("Training Samples", f"{n_train:,}")

            # ── ESTIMATE FORM ─────────────────────────────────────────────
            st.markdown('<div class="section-header">Get a Price Estimate</div>', unsafe_allow_html=True)

            with st.form("estimate_form"):
                fc1, fc2, fc3 = st.columns(3)
                with fc1:
                    f_sqft = st.number_input("Area (sqft)", min_value=100, value=1000)
                    f_bed = st.selectbox(
                        "Bedrooms",
                        ["1 RK", "1 BHK", "2 BHK", "3 BHK", "4 BHK", "5 BHK"],
                        index=1,
                    )
                    f_bath = st.number_input("Bathrooms", min_value=1, value=2)
                with fc2:
                    f_floor = st.number_input("Floor", min_value=1, value=5)
                    f_total_floors = st.number_input("Total Floors", min_value=1, value=10)
                    f_parking = st.number_input("Parking Spots", min_value=0, value=1)
                with fc3:
                    f_age = st.number_input("Age (years)", min_value=0, value=5)
                    f_metro = st.number_input("Metro Distance (km)", min_value=0.0, value=2.0, step=0.1)
                    f_furnish = (
                        st.selectbox("Furnishing", df["furnishing"].dropna().unique().tolist())
                        if "furnishing" in df.columns
                        else "Unfurnished"
                    )

                fc4, fc5 = st.columns(2)
                with fc4:
                    f_facing = (
                        st.selectbox("Facing", df["facing"].dropna().unique().tolist())
                        if "facing" in df.columns
                        else "North"
                    )
                with fc5:
                    f_location = (
                        st.selectbox("Location", df["location"].dropna().unique().tolist())
                        if "location" in df.columns
                        else None
                    )

                submit = st.form_submit_button("💡 Estimate Price")

            if submit:
                input_dict = {
                    "sqft": f_sqft,
                    "bedrooms": bedroom_to_num(f_bed),
                    "bathrooms": f_bath,
                    "floor": f_floor,
                    "total_floors": f_total_floors,
                    "age_years": f_age,
                    "parking": f_parking,
                    "metro_distance_km": f_metro,
                }
                if "furnishing" in df.columns:
                    input_dict["furnishing"] = f_furnish
                if "facing" in df.columns:
                    input_dict["facing"] = f_facing
                if f_location and "location" in df.columns:
                    input_dict["location"] = f_location
                    if "state" in df.columns:
                        s = df[df["location"] == f_location]["state"].dropna()
                        input_dict["state"] = s.iloc[0] if len(s) > 0 else df["state"].iloc[0]

                # Build one-hot features inline
                try:
                    cat_features_present = [
                        c for c in ["state", "location", "furnishing", "facing"]
                        if c in df.columns
                    ]
                    row_df = pd.DataFrame([input_dict])
                    if cat_features_present:
                        dummies = pd.get_dummies(row_df, columns=cat_features_present, drop_first=True)
                    else:
                        dummies = row_df.copy()

                    for col in analyzer.features:
                        if col not in dummies.columns:
                            dummies[col] = 0
                    dummies = dummies[analyzer.features]
                    scaled = analyzer.scaler.transform(dummies)
                    price_est = float(analyzer.model.predict(scaled)[0])

                    st.markdown(
                        f"<div style='background:linear-gradient(135deg,#1a2236,#131929);"
                        f"border:1px solid rgba(240,165,0,0.3);border-radius:16px;"
                        f"padding:2rem;text-align:center;margin-top:1rem'>"
                        f"<div style='color:#7a859a;font-size:0.85rem;text-transform:uppercase;"
                        f"letter-spacing:0.1em'>Estimated Market Price</div>"
                        f"<div style='font-family:Syne,sans-serif;font-size:3rem;font-weight:800;"
                        f"background:linear-gradient(135deg,#f0a500,#e05c5c);"
                        f"-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
                        f"background-clip:text'>₹{price_est:,.2f}L</div>"
                        f"<div style='color:#7a859a;font-size:0.8rem'>via {analyzer.model_type}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Prediction error: {e}")
        else:
            st.info("👆 Train the model above to unlock price estimation.")


if __name__ == "__main__":
    main()

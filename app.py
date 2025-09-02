# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
from PIL import Image, ImageOps
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ReportLab (PDF). If unavailable, PDF features will be disabled.
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    HAVE_RPT = True
except Exception:
    HAVE_RPT = False

# ---------------- Page config + dark theme CSS ----------------
st.set_page_config(page_title="⚡ Smart Energy Grid Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #071425; color: #e6eef8; }
    .insight-card { background: #0b1a2b; border: 1px solid rgba(255,255,255,0.04);
                    border-radius:12px; padding:12px; color:#e6eef8; }
    .insight-title { font-weight:700; font-size:15px; }
    .insight-sub { color:#93b1d6; font-size:12px; margin-top:4px; }
    .insight-big { font-size:18px; font-weight:700; margin-top:8px; color:#ffffff; }
    .small-muted { color:#8b9bb6; font-size:12px; }
    .center { text-align:center; }
    </style>
    """, unsafe_allow_html=True
)
plt.style.use("dark_background")

# ---------------- Header ----------------
st.markdown("<h1 style='text-align:center; color:#7dd3fc;'>⚡ Smart Energy Grid Dashboard</h1>", unsafe_allow_html=True)
st.caption("Upload a CSV (center) or use the sample dataset. App accepts full set of sources or any subset (e.g., Wind only).")

# ---------------- Data upload (MAIN PAGE) ----------------
st.markdown("## 📂 Upload CSV (optional)")
uploaded = st.file_uploader("Upload a CSV with a date/time column (optional) and numeric generation columns (e.g., Geothermal, Hydro, Solar, Wind).", type=["csv"])

def sample_data(n=120):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Month": pd.date_range("2023-01-01", periods=n, freq="ME"),
        "Geothermal": rng.integers(250, 400, size=n),
        "Hydro": rng.integers(120, 300, size=n),
        "Solar": rng.integers(40, 140, size=n),
        "Wind": rng.integers(70, 200, size=n),
    })

if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
        st.success("CSV loaded successfully.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()
else:
    df_raw = sample_data(120)
    st.info("Using sample dataset (upload a CSV to use your own data).")

# Use copy for safety
df = df_raw.copy()

# ---------------- Detect time column / index ----------------
date_candidates = [c for c in df.columns if c.lower() in ("month", "date", "datetime", "time")]
time_col = None
if date_candidates:
    time_col = date_candidates[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    if df[time_col].isna().all():
        time_col = None
if time_col is None:
    df["IndexTime"] = np.arange(len(df))
    time_col = "IndexTime"

# ---------------- Detect numeric plant columns ----------------
exclude = {time_col}
plant_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
if not plant_cols:
    st.error("No numeric plant columns found. Ensure your CSV has numeric generation columns like Geothermal, Hydro, Solar, Wind.")
    st.stop()

# ---------------- CSV Guidelines expander ----------------
with st.expander("CSV Upload Guidelines (click to expand)"):
    st.markdown("""
    **Recommended format**
    - Optional time column named `Month`, `Date`, `Datetime`, or `Time` (ISO-like dates).
    - Numeric columns for generation: e.g. `Geothermal`, `Hydro`, `Solar`, `Wind` (any subset ok).
    - Example:
      ```
      Month,Geothermal,Hydro,Solar,Wind
      2023-01-31,300,200,45,120
      2023-02-28,310,190,50,130
      ```
    **Tips**
    - Remove thousands separators (commas) from numbers.
    - Ensure numeric columns contain numbers only.
    - You can upload files with only one plant (e.g., `Month,Wind`) — the app will still analyze and forecast it.
    """)
    st.write("Preview (first 5 rows):")
    st.dataframe(df.head().fillna("").astype(str))

# ---------------- Insights summary ----------------
st.markdown("## 🔎 Insights Summary")

totals = df[plant_cols].sum()
total_all = totals.sum() if totals.sum() != 0 else 1.0
perc = (totals / total_all * 100).round(1)

# Try to load optional site icons (earth.png for geothermal etc.)
icon_files = {
    "Geothermal": "earth.png",
    "Solar": "solar.png",
    "Wind": "wind.png",
    "Hydro": "hydro.png",
}
site_icons = {}
for k, f in icon_files.items():
    try:
        img = Image.open(f).convert("RGBA")
        img = ImageOps.fit(img, (64, 64), Image.LANCZOS)
        site_icons[k] = img
    except Exception:
        site_icons[k] = None

taglines = {
    "Geothermal": "Stable baseload",
    "Solar": "Daylight-dependent",
    "Wind": "Variable gust patterns",
    "Hydro": "Seasonal inflows",
}

preferred = ["Geothermal", "Solar", "Wind", "Hydro"]
available_order = [p for p in preferred if p in plant_cols] + [p for p in plant_cols if p not in preferred]

card_count = min(4, len(available_order))
cols_cards = st.columns(card_count)
for i in range(card_count):
    p = available_order[i]
    with cols_cards[i]:
        if site_icons.get(p) is not None:
            st.image(site_icons[p], width=48)
        else:
            emoji_map = {"Geothermal":"🪨","Solar":"☀️","Wind":"🌬️","Hydro":"💧"}
            st.markdown(f"<div style='font-size:28px'>{emoji_map.get(p,'')}</div>", unsafe_allow_html=True)

        rows = len(df)
        missing = int(df[p].isna().sum())
        mean_val = float(df[p].mean())

        st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='insight-title'>{p}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='insight-sub'>{taglines.get(p,'')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='insight-big'>{float(perc.get(p,0.0)):.1f}%</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>Rows: <b>{rows}</b> • Avg: <b>{mean_val:.1f} MW</b></div>", unsafe_allow_html=True)
        if missing > 0:
            st.markdown(f"<div class='small-muted'>Missing values: <b>{missing}</b> — consider cleaning.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Energy mix bar ----------------
st.markdown("## 📊 Energy Mix (Percent Contribution)")
fig_mix, ax_mix = plt.subplots(figsize=(6, 2.8))
labels = list(perc.index)
values = list(perc.values)
color_map = {"Geothermal":"#f97316", "Solar":"#fde047", "Wind":"#60a5fa", "Hydro":"#34d399"}
bar_colors = [color_map.get(l, "#9fb3d6") for l in labels]
ax_mix.bar(labels, values, color=bar_colors)
ax_mix.set_ylabel("Percent (%)")
ax_mix.set_ylim(0, max(100, max(values) + 10))
for idx, v in enumerate(values):
    ax_mix.text(idx, v + 1, f"{v:.1f}%", ha="center", fontsize=9, color="white")
fig_mix.tight_layout()
st.pyplot(fig_mix)

# ---------------- Compact source trends ----------------
st.markdown("## 🔋 Source Trends (compact)")
cols_per_row = 2
rows_r = (len(plant_cols) + cols_per_row - 1) // cols_per_row
palette_default = ["#a78bfa", "#fb7185", "#60a5fa", "#34d399"]
for r in range(rows_r):
    row_cols = st.columns(cols_per_row)
    for c in range(cols_per_row):
        idx = r * cols_per_row + c
        if idx >= len(plant_cols):
            continue
        p = plant_cols[idx]
        fig, ax = plt.subplots(figsize=(5, 2.2))
        x_axis = df[time_col]
        color = color_map.get(p, palette_default[idx % len(palette_default)])
        ax.plot(x_axis, df[p], label=p, color=color, linewidth=1.3)
        ax.set_title(p, color="#dbeafe")
        ax.tick_params(colors="#9fb3d6")
        ax.set_xlabel("")
        ax.set_ylabel("MW")
        ax.legend(loc="upper left", fontsize=8)
        row_cols[c].pyplot(fig)

# ---------------- Forecast controls (sidebar) ----------------
st.sidebar.header("⚙️ Forecast Controls")
model_name = st.sidebar.selectbox("Model", ["Linear Regression", "Ridge", "Lasso"])
max_horizon = max(1, min(24, len(df)//4))
horizon = st.sidebar.number_input("Holdout horizon (periods)", min_value=1, max_value=max_horizon, value=min(12, max_horizon))
plant_options = ["All"] + plant_cols
plants_pick = st.sidebar.multiselect("Plants to Forecast", options=plant_options, default=[plant_cols[0]])
if "All" in plants_pick:
    selected_plants = plant_cols.copy()
else:
    selected_plants = [p for p in plants_pick if p in plant_cols]
run_forecast = st.sidebar.button("Run Forecast")

# ---------------- Helpers: holdout forecasting ----------------
def fit_and_forecast_holdout(series: pd.Series, model_choice: str, horizon: int):
    n = len(series)
    train_n = max(1, n - horizon)
    X_train = np.arange(train_n).reshape(-1, 1)
    y_train = series.iloc[:train_n].to_numpy(dtype=float)
    X_test = np.arange(train_n, n).reshape(-1, 1)
    y_test = series.iloc[train_n:].to_numpy(dtype=float)

    if model_choice == "Linear Regression":
        mdl = LinearRegression()
    elif model_choice == "Ridge":
        mdl = Ridge()
    else:
        mdl = Lasso()

    mdl.fit(X_train, y_train)
    preds_train = mdl.predict(np.arange(0, train_n).reshape(-1, 1))
    preds_holdout = mdl.predict(X_test) if len(X_test) > 0 else np.array([])
    preds_full = np.concatenate([preds_train, preds_holdout]) if len(preds_holdout) > 0 else preds_train
    r2 = r2_score(y_test, preds_holdout) if len(y_test) > 0 else float('nan')
    mse = mean_squared_error(y_test, preds_holdout) if len(y_test) > 0 else float('nan')
    mae = mean_absolute_error(y_test, preds_holdout) if len(y_test) > 0 else float('nan')
    return preds_full, r2, mse, mae, train_n

# ---------------- Run forecast and plotting ----------------
forecast_summary_lines = []
chart_images = []  # list of (name, BytesIO) for embedding in PDF

if run_forecast and selected_plants:
    st.markdown("## 🔮 Forecast Results (holdout evaluation)")
    fig_f, ax_f = plt.subplots(figsize=(9, 3.6))

    x_axis = df[time_col]
    if not pd.api.types.is_datetime64_any_dtype(x_axis):
        x_plot = np.arange(len(df))
    else:
        x_plot = x_axis

    r2s = {}
    for i, p in enumerate(selected_plants):
        series = df[p].astype(float).copy().reset_index(drop=True)
        preds_full, r2, mse, mae, train_n = fit_and_forecast_holdout(series, model_name, int(horizon))
        color = color_map.get(p, palette_default[i % len(palette_default)])

        # plot actual
        ax_f.plot(x_plot, series, label=f"{p} Actual", color=color, linewidth=1.3)
        # plot fit and holdout
        if train_n > 0:
            ax_f.plot(x_plot[:train_n], preds_full[:train_n], label=f"{p} Fit", color=color, linewidth=1.0, alpha=0.7)
        if len(preds_full) > train_n:
            ax_f.plot(x_plot[train_n:], preds_full[train_n:], label=f"{p} Forecast", linestyle="--", color=color, linewidth=1.6)

        forecast_summary_lines.append(f"{p}: R²={r2:.3f}, MSE={mse:.2f}, MAE={mae:.2f} (holdout)")
        r2s[p] = r2

        # save per-plant chart image to embed in PDF
        buf = BytesIO()
        fig_p, ax_p = plt.subplots(figsize=(6, 2.6))
        ax_p.plot(x_plot, series, label="Actual", color=color)
        if train_n > 0:
            ax_p.plot(x_plot[:train_n], preds_full[:train_n], label="Fit", color=color, alpha=0.7)
        if len(preds_full) > train_n:
            ax_p.plot(x_plot[train_n:], preds_full[train_n:], label="Forecast", linestyle="--", color=color)
        ax_p.set_title(f"{p} — {model_name} (holdout)")
        ax_p.legend(loc="upper left", fontsize=8)
        fig_p.tight_layout()
        fig_p.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig_p)
        buf.seek(0)
        chart_images.append((p, buf))

    ax_f.set_title(f"Forecast: {', '.join(selected_plants)} — {model_name}", color="#dbeafe")
    ax_f.set_xlabel("Time")
    ax_f.set_ylabel("MW")
    ax_f.legend(ncol=2, fontsize=9)
    ax_f.tick_params(colors="#9fb3d6")
    fig_f.tight_layout()
    st.pyplot(fig_f)

    # written explanation
    st.markdown("### 📝 Summary Explanation")
    expl = []
    if "Geothermal" in selected_plants:
        expl.append("- **Geothermal**: stable baseload with low variance — reliable contributor.")
    if "Solar" in selected_plants:
        expl.append("- **Solar**: daylight-driven; expect midday/seasonal peaks.")
    if "Wind" in selected_plants:
        expl.append("- **Wind**: variable and gust-driven; higher short-term fluctuations.")
    if "Hydro" in selected_plants:
        expl.append("- **Hydro**: influenced by seasonal inflows and reservoir management.")
    if not expl:
        expl.append("- Forecast shows general trends over sampled period.")
    st.markdown("\n".join(expl))

    # Model comparison chart (avg R2)
    models_list = ["Linear Regression", "Ridge", "Lasso"]
    comp_scores = []
    for m in models_list:
        scores = []
        for p in selected_plants:
            s = df[p].astype(float).copy().reset_index(drop=True)
            _, r2m, _, _, _ = fit_and_forecast_holdout(s, m, int(horizon))
            scores.append(r2m if not np.isnan(r2m) else 0.0)
        avg = float(np.mean(scores)) if scores else 0.0
        comp_scores.append({"Model": m, "Avg_R2": round(avg, 3)})
    comp_df = pd.DataFrame(comp_scores)

    st.markdown("### 🔍 Model Comparison (avg R² across selected plants)")
    fig_bar, ax_bar = plt.subplots(figsize=(6, 2.6))
    ax_bar.bar(comp_df["Model"], comp_df["Avg_R2"], color=["#60a5fa", "#f97316", "#34d399"])
    ax_bar.set_ylim(0, 1)
    for i, v in enumerate(comp_df["Avg_R2"]):
        ax_bar.text(i, v + 0.02, f"{v:.3f}", ha="center", color="white")
    fig_bar.tight_layout()
    st.pyplot(fig_bar)

    # save model comparison chart for PDF
    buf_bar = BytesIO()
    fig_bar.savefig(buf_bar, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig_bar)
    buf_bar.seek(0)
    chart_images.append(("model_comparison", buf_bar))

# ---------------- Build PDF in-memory and download ----------------
st.markdown("---")
st.markdown("<div class='center'><h3>📥 Download Report (PDF)</h3></div>", unsafe_allow_html=True)

def build_pdf_bytes(insights_perc: pd.Series, chart_imgs, forecast_lines=None):
    if not HAVE_RPT:
        return None
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Smart Energy Grid — Professional Report")
    c.setFont("Helvetica", 9)
    c.drawString(50, height - 65, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    # Insights
    y = height - 95
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Insights Summary")
    y -= 16
    c.setFont("Helvetica", 10)
    for name in available_order:
        if name in insights_perc.index:
            c.drawString(60, y, f"{name}: {float(insights_perc[name]):.1f}% contribution (sampled)")
            y -= 12

    # Forecast metrics
    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Forecast Metrics (holdout)")
    y -= 16
    c.setFont("Helvetica", 10)
    if forecast_lines:
        for line in forecast_lines:
            if y < 80:
                c.showPage()
                y = height - 60
            c.drawString(60, y, line)
            y -= 12
    else:
        c.drawString(60, y, "No forecast run in this session.")
        y -= 12

    # Insert chart images (each on its own page)
    for name, img_buf in chart_imgs:
        c.showPage()
        try:
            img_reader = ImageReader(img_buf)
            iw, ih = img_reader.getSize()
            max_w = width - 100
            max_h = height - 120
            ratio = min(max_w / iw, max_h / ih, 1.0)
            draw_w, draw_h = iw * ratio, ih * ratio
            x = (width - draw_w) / 2
            y_img = (height - draw_h) / 2
            c.drawImage(img_reader, x, y_img, width=draw_w, height=draw_h)
        except Exception:
            c.setFont("Helvetica", 10)
            c.drawString(50, height - 100, f"Could not embed image {name}")

    c.save()
    buffer.seek(0)
    return buffer

pdf_buffer = None
if HAVE_RPT:
    charts_for_pdf = chart_images.copy() if chart_images else []
    pdf_buffer = build_pdf_bytes(perc, charts_for_pdf, forecast_summary_lines if forecast_summary_lines else None)

if HAVE_RPT and pdf_buffer is not None:
    st.download_button(
        label="⬇️ Save PDF Report",
        data=pdf_buffer,
        file_name="smart_energy_grid_report.pdf",
        mime="application/pdf",
    )
elif not HAVE_RPT:
    st.warning("`reportlab` not installed. Install with: pip install reportlab to enable PDF export.")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "<div class='center small-muted'>Prototype by Simon Wanyoike • contact: allinmer57@gmail.com<br>Harnessing data to power Kenya's clean energy future</div>",
    unsafe_allow_html=True
)

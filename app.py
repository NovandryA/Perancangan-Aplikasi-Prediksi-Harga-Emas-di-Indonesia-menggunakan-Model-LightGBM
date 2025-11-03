import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import joblib
import json 
from pathlib import Path
from io import BytesIO
import plotly.graph_objects as go
import re

# validasi tanggal
DATASET_START = date(2014, 1, 1)
DATASET_END   = date(2024, 12, 31)
MIN_ALLOWED   = DATASET_END + timedelta(days=1)

MODELS_DIR = Path("models")
META_PATH  = MODELS_DIR / "model_meta.json"
METRICS_CSV = MODELS_DIR / "metrics_all_horizons.csv"


VALID_PATH = Path("Dataset_HargaEmas_2025.xlsx")
df_valid = None
if VALID_PATH.exists():
    df_valid = pd.read_excel(VALID_PATH)
    df_valid.columns = [c.strip().replace(" ", "_") for c in df_valid.columns]
    df_valid["Date"] = pd.to_datetime(df_valid["Date"])


# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Harga Emas — LightGBM",
    page_icon="🪙",
    layout="wide",
)


# CSS Custom
CUSTOM_CSS = """
<style>
    /* Global tweaks */
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    /* Metric-style cards */
    .metric-card {
        border: 1px solid rgba(49,51,63,0.2);
        border-radius: 16px; padding: 1.1rem 1.25rem; margin-bottom: 0.75rem;
        background: rgba(250, 250, 250, 0.65);
    }
    .metric-title { font-size: 0.9rem; opacity: 0.75; margin-bottom: 0.25rem; }
    .metric-value { font-weight: 700; font-size: 1.25rem; }
    /* Subtle section card */
    .section-card {
        border: 1px dashed rgba(49,51,63,0.2);
        border-radius: 12px; padding: 1rem; background: rgba(255,255,255,0.5);
    }
    /* Buttons spacing in forms */
    .stButton > button { border-radius: 10px; font-weight: 600; padding: 0.5rem 1rem; }

</style>
"""
st.markdown("""
<style>
/* container tombol kita sendiri */
.button {
    background-color: #FF991C !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.2rem !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25) !important;
    cursor: pointer !important;
}

/* efek hover */
.button:hover {
    background-color: #e07f00 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)



# Helper function
def kpi_card(title: str, value: str, help_text: str | None = None):
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-title">{title}</div>', unsafe_allow_html=True)
        if help_text:
            st.markdown(
                f'<div class="metric-value" title="{help_text}">{value}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f'<div class="metric-value">{value}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def placeholder_forecast_df(start: date, days: int = 7) -> pd.DataFrame:
    
    dates = pd.date_range(start=start + timedelta(days=1), periods=days, freq="D")
    # 
    base = 1_000_000 
    noise = np.cumsum(np.random.normal(scale=3000, size=days))
    trend = np.linspace(0, 20_000, days)
    y = base + trend + noise
    df = pd.DataFrame({"Date": dates, "Harga (Preview)": y})
    return df


# Helpers untuk model
def load_meta(meta_path: Path) -> dict:
    with open(meta_path, "r") as f:
        return json.load(f)
    
def _parse_bi_to_percent(x) -> float | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().replace('%', '').replace(' ', '')

    
    if re.fullmatch(r'\d+', s):
        v = int(s)
        if 50 <= v < 1000:   
            return v / 100.0
        return float(v)    

    # ada koma/titik
    if ',' in s and '.' not in s:
        
        s = s.replace('.', '')           
        try:
            v = float(s.replace(',', '.'))
        except ValueError:
            return None
    elif '.' in s and ',' not in s:
        #
        try:
            v = float(s)
        except ValueError:
            return None
    else:
        
        last_comma = s.rfind(',')
        last_dot   = s.rfind('.')
        if last_comma > last_dot:
            
            s = s.replace('.', '')
            try:
                v = float(s.replace(',', '.'))
            except ValueError:
                return None
        else:
            
            s = s.replace(',', '')
            try:
                v = float(s)
            except ValueError:
                return None

   
    return v * 100.0 if v is not None and v <= 1.0 else v

def normalize_bi_rate_to_str(series: pd.Series) -> pd.Series:
    pct = series.apply(_parse_bi_to_percent)

    return pct.map(lambda v: (f"{v:.2f}%".replace('.', ',')) if pd.notna(v) else None)

def make_xlsx_bytes(df: pd.DataFrame) -> bytes:
    
    if 'date' in df.columns:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter", datetime_format='yyyy-mm-dd', date_format='yyyy-mm-dd') as writer:
        df.to_excel(writer, sheet_name="HargaEmas2025", index=False)
    bio.seek(0)
    return bio.read()


def load_model_for_horizon(h: int):
    meta = load_meta(META_PATH)
    target_kind = meta[f"Delta{h}"]["target"]            
    feat_cols   = meta[f"Delta{h}"]["features"]          
    model_path  = MODELS_DIR / f"Model_LightGBM_GOSS_Delta{h}_{target_kind}.pkl"
    model = joblib.load(model_path)
    return model, feat_cols, target_kind

def reconstruct_next(prev_level: float, pred: float, target_kind: str):
    if target_kind == "log":
        nxt = float(np.exp(pred) * prev_level)
        delta = nxt - prev_level
    else:
        nxt = float(prev_level + pred)
        delta = pred
    return nxt, delta

def reconstruct_next(prev_level: float, pred: float, target_kind: str):
    """
    Rekonstruksi level dari prediksi target per horizon.
    - target_kind == 'log'   : pred = rN = log(P_{t+N}/P_{t+N-1})
    - target_kind == 'delta' : pred = ΔN = P_{t+N} - P_{t+N-1}
    """
    if target_kind == "log":
        next_level = float(np.exp(pred) * prev_level)
        delta = next_level - prev_level
    else:
        next_level = float(prev_level + pred)
        delta = pred
    return next_level, delta

# Fungsi membatasi input user

# def validate_user_input(tanggal_input: date,
#                         harga_emas_input: float,
#                         usd_sell_input: float,
#                         usd_buy_input: float,
#                         bi_rate_input: float,
#                         df_valid: pd.DataFrame):

#     df_check = df_valid.copy()
#     df_check["Date"] = pd.to_datetime(df_check["Date"]).dt.date

#     row = df_check[df_check["Date"] == tanggal_input]
#     if row.empty:
#         return (
#             False,
#             "Tanggal tidak ditemukan di data referensi.",
#             {}
#         )

#     ref_gold      = float(row["Gold_Price"].iloc[0])
#     ref_usd_sell  = float(row["USD_Sell_Rate"].iloc[0])
#     ref_usd_buy   = float(row["USD_Buy_Rate"].iloc[0])
#     ref_bi_rate   = float(row["BI_Rate"].iloc[0])

#     same_gold     = round(harga_emas_input, 2) == round(ref_gold, 2)
#     same_usd_sell = round(usd_sell_input,  2) == round(ref_usd_sell, 2)
#     same_usd_buy  = round(usd_buy_input,   2) == round(ref_usd_buy, 2)
#     same_bi_rate  = round(bi_rate_input,   4) == round(ref_bi_rate * 100, 4) if ref_bi_rate < 1 else round(bi_rate_input, 2) == round(ref_bi_rate, 2)

#     mismatches = {}

#     if not same_gold:
#         mismatches["Harga Emas (IDR/gram)"] = {
#             "input": f"{harga_emas_input:,.2f}",
#             "seharusnya": f"{ref_gold:,.2f}",
#         }
#     if not same_usd_sell:
#         mismatches["Kurs USD/IDR — Jual"] = {
#             "input": f"{usd_sell_input:,.2f}",
#             "seharusnya": f"{ref_usd_sell:,.2f}",
#         }
#     if not same_usd_buy:
#         mismatches["Kurs USD/IDR — Beli"] = {
#             "input": f"{usd_buy_input:,.2f}",
#             "seharusnya": f"{ref_usd_buy:,.2f}",
#         }
#     if not same_bi_rate:
#         # Konversi ke persen (5,75%)
#         input_percent = bi_rate_input
#         ref_percent = ref_bi_rate * 100 if ref_bi_rate < 1 else ref_bi_rate
#         mismatches["BI-Rate (%)"] = {
#             "input": f"{input_percent:.2f}%",
#             "seharusnya": f"{ref_percent:.2f}%"
#         }

#     if len(mismatches) > 0:
#         return (
#             False,
#             "Data tidak valid. Ada nilai yang tidak sesuai dengan kondisi aktual hari tersebut.",
#             mismatches
#         )

#     return (True, None, {})
def validate_user_input(tanggal_input: date,
                        harga_emas_input: float,
                        usd_sell_input: float,
                        usd_buy_input: float,
                        bi_rate_input: float,
                        df_valid: pd.DataFrame):
    """
    Memvalidasi input user berdasarkan data referensi (df_valid).
    Mengembalikan (True, None, {}) jika semua valid.
    Jika ada mismatch, kembalikan (False, pesan, dict detail).
    """

    df_check = df_valid.copy()
    df_check["Date"] = pd.to_datetime(df_check["Date"]).dt.date

    row = df_check[df_check["Date"] == tanggal_input]
    if row.empty:
        return (
            False,
            "Tanggal tidak ditemukan di data referensi.",
            {}
        )

    # Ambil nilai referensi
    ref_gold      = float(row["Gold_Price"].iloc[0])
    ref_usd_sell  = float(row["USD_Sell_Rate"].iloc[0])
    ref_usd_buy   = float(row["USD_Buy_Rate"].iloc[0])
    ref_bi_rate   = float(row["BI_Rate"].iloc[0])

    # Cek kesamaan nilai (toleransi dua desimal)
    same_gold     = round(harga_emas_input, 2) == round(ref_gold, 2)
    same_usd_sell = round(usd_sell_input,  2) == round(ref_usd_sell, 2)
    same_usd_buy  = round(usd_buy_input,   2) == round(ref_usd_buy, 2)
    same_bi_rate  = round(bi_rate_input,   4) == round(ref_bi_rate * 100, 4) if ref_bi_rate < 1 else round(bi_rate_input, 2) == round(ref_bi_rate, 2)

    mismatches = {}

    # Format tanpa koma ribuan
    def fmt_number(val):
        return f"{val:.2f}".replace(",", "")

    if not same_gold:
        mismatches["Harga Emas (IDR/gram)"] = {
            "input": fmt_number(harga_emas_input),
            "seharusnya": fmt_number(ref_gold),
        }
    if not same_usd_sell:
        mismatches["Kurs USD/IDR — Jual"] = {
            "input": fmt_number(usd_sell_input),
            "seharusnya": fmt_number(ref_usd_sell),
        }
    if not same_usd_buy:
        mismatches["Kurs USD/IDR — Beli"] = {
            "input": fmt_number(usd_buy_input),
            "seharusnya": fmt_number(ref_usd_buy),
        }
    if not same_bi_rate:
        input_percent = bi_rate_input
        ref_percent = ref_bi_rate * 100 if ref_bi_rate < 1 else ref_bi_rate
        mismatches["BI-Rate (%)"] = {
            "input": f"{input_percent:.2f}%",
            "seharusnya": f"{ref_percent:.2f}%"
        }

    if len(mismatches) > 0:
        return (
            False,
            "Data tidak valid. Ada nilai yang tidak sesuai dengan kondisi aktual hari tersebut.",
            mismatches
        )

    return (True, None, {})

# Sidebar Navigasi
with st.sidebar:
    st.header("🪙 Prediksi Harga Emas Menggunakan LightGBM")
    nav = st.radio("Navigasi", ["Home", "Prediction", "About"], index=0)

    st.divider()

    with open("Manual Book.pdf", "rb") as f:
        pdf_data = f.read()
    st.download_button(
        label="📘 Download Manual Book",
        data=pdf_data,
        file_name="Manual Book.pdf",
        mime="application/pdf",
        use_container_width=True)


# Halaman: Home
def render_home():
    left, right = st.columns([1.1, 1])

    with left:
        st.title("Perancangan Aplikasi Prediksi Harga Emas")
        st.write(
            "Aplikasi berbasis **Streamlit** untuk memprediksi harga emas di Indonesia menggunakan model **LightGBM**"
        )

        st.subheader("🏦 Makroekonomi dan Faktor-Faktor yang Mempengaruhi Harga Emas")
        st.markdown(
            """
            Pergerakan harga emas di Indonesia sangat dipengaruhi oleh kondisi makroekonomi, yaitu faktor-faktor ekonomi berskala besar yang mencerminkan kesehatan perekonomian nasional. 
            Beberapa di antaranya mencakup tingkat suku bunga, inflasi, kurs valuta asing, serta stabilitas ekonomi global. 
            Ketika kondisi ekonomi tidak menentu, emas sering dianggap sebagai aset aman (safe haven) sehingga permintaannya meningkat dan harga cenderung naik.
            """
        )

        st.subheader("💵 Kurs USD/IDR")
        st.markdown(
            """
            Kurs USD/IDR menunjukkan nilai tukar antara dolar Amerika Serikat dan rupiah Indonesia. 
            Perubahan nilai tukar ini berpengaruh langsung terhadap harga emas domestik karena harga emas dunia dipatok dalam dolar AS. 
            Ketika rupiah melemah terhadap dolar, harga emas dalam rupiah biasanya naik karena biaya konversi menjadi lebih mahal, meskipun harga emas dunia tidak berubah.
            """
        )

        st.subheader("📈 BI-Rate")
        st.markdown(
            """
            BI-Rate adalah suku bunga acuan yang ditetapkan oleh Bank Indonesia untuk mengendalikan inflasi dan menjaga stabilitas moneter. Kenaikan BI-Rate biasanya menekan harga emas karena investor cenderung beralih ke instrumen keuangan yang menawarkan bunga lebih tinggi.
            Sebaliknya, ketika BI-Rate turun, minat terhadap emas meningkat karena biaya peluang menyimpan emas menjadi lebih rendah.
            """
        )

        st.subheader("🤖 LightGBM")
        st.markdown(
            """
            Dalam aplikasi ini digunakan algoritma Light Gradient Boosting Machine (LightGBM) — sebuah metode pembelajaran mesin (machine learning) yang dirancang untuk melakukan prediksi secara cepat dan efisien. LightGBM bekerja dengan membangun serangkaian decision tree secara bertahap, di mana setiap pohon baru berfokus memperbaiki kesalahan dari pohon sebelumnya. 
            Dengan teknik optimasi seperti Gradient-based One-Side Sampling (GOSS) LightGBM mampu memberikan hasil prediksi yang akurat meskipun data berukuran besar.
            """
        )

    st.divider()

    st.subheader("Cara Kerja")
    cols = st.columns(4)
    steps = [
        ("Input Variabel", "Masukkan 4 variabel harian: harga emas, kurs jual, kurs beli, BI‑Rate."),
        ("Validasi", "Sistem cek format, range, dan kelengkapan input."),
        ("Prediksi", "Model LightGBM menghasilkan proyeksi 7 hari ke depan."),
        ("Visualisasi", "Hasil tampil sebagai grafik perbandingan antara harga prediksi dan harga aktual."),
    ]
    for i, (t, d) in enumerate(steps):
        with cols[i]:
            st.markdown(f"**{t}**")
            st.caption(d)

    st.divider()

    st.markdown (
        """Sumber data dari aplikasi perancangan ini adalah: <https://www.bi.go.id/id/statistik/informasi-kurs/transaksi-bi/Default.aspx> (Kurs USD/IDR),
        <https://www.bi.go.id/id/statistik/indikator/bi-rate.aspx> (BI-Rate), dan <https://harga-emas.org/> (Harga Emas)."""
    )


# Halaman: Prediction
def render_prediction():
    st.title("Prediction")
    st.caption("Masukkan data hari ini untuk memprediksi harga emas hingga 7 hari ke depan.")
    st.write(
            "Note: Agar Model berjalan dengan baik, pastikan input variabel merupakan data yang valid."
        )
    
    st.divider()
    st.subheader("Unduh Dataset Harga Emas 2025")

    df_download = None
    try:
        if "df_valid" in globals() and isinstance(df_valid, pd.DataFrame) and not df_valid.empty:
            df_download = df_valid.copy()
        else:
          
            df_download = pd.read_excel("Dataset_HargaEmas_2025.xlsx")  
    except Exception as e:
        st.warning(f"Tidak bisa memuat dataset 2025: {e}")

    if df_download is not None and not df_download.empty:
       
        rename_map = {}
        if "Tanggal" in df_download.columns:   rename_map["Tanggal"]    = "date"
        if "Gold_Price" in df_download.columns:rename_map["Gold_Price"] = "gold_price"
        
        for c in df_download.columns:
            lc = c.lower().replace(' ', '').replace('_','').replace('-','')
            if lc in ("birate","biratepersen","biratepercent","biratepersentase"):
                rename_map[c] = "bi_rate"
        df_download = df_download.rename(columns=rename_map)

    
        if "bi_rate" in df_download.columns:
            df_download["bi_rate"] = normalize_bi_rate_to_str(df_download["bi_rate"])

        
        ordered_cols = [c for c in [
            "date", "gold_price", "USD_Sell_Rate", "USD_Buy_Rate", "bi_rate"
        ] if c in df_download.columns]
        other_cols = [c for c in df_download.columns if c not in ordered_cols]
        final_df = df_download[ordered_cols + other_cols]

        st.download_button(
            label="⬇️ Download data Harga Emas 2025 (1 Januari 2025 - 31 Oktober 2025) ",
            data=make_xlsx_bytes(final_df),
            file_name="harga_emas_2025.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
        st.info("Dataset 2025 belum tersedia.")


    # ----- FORM -----
    with st.form(key="predict_form", clear_on_submit=False):
        st.subheader("Input Variabel (Harian)")
        c1, c2 = st.columns(2)
        with c1:
            tanggal = st.date_input(
                "Tanggal Data",
                value=min(date.today(), date(2025, 10, 24)),  
                min_value=MIN_ALLOWED,                        
                max_value=date(2025, 10, 24),                 
                help="Tanggal harus antara 1 Januari 2025 dan 24 Oktober 2025."
            )
            harga_emas = st.number_input("Harga Emas (IDR/gram)", min_value=0.0, step=1000.0, format="%.2f")
            usd_sell   = st.number_input("Kurs USD/IDR — Jual",   min_value=0.0, step=1.0,   format="%.2f")
        with c2:
            usd_buy    = st.number_input("Kurs USD/IDR — Beli",   min_value=0.0, step=1.0,   format="%.2f")
            bi_rate    = st.number_input("BI-Rate (%)",           min_value=0.0, max_value=25.0, step=0.05, format="%.2f")
            horizon    = st.slider("Horizon Prediksi (hari)", 1, 7, 7)

        submitted = st.form_submit_button("Prediksi")

    if not submitted:
        return

    # Validasi tambahan 
    if DATASET_START <= tanggal <= DATASET_END:
        st.error("Tanggal harus di luar periode dataset (pilih setelah 31 Desember 2024).")
        return
    
    is_ok, err_msg, mismatches = validate_user_input(
        tanggal_input = tanggal,
        harga_emas_input = harga_emas,
        usd_sell_input = usd_sell,
        usd_buy_input = usd_buy,
        bi_rate_input = bi_rate,
        df_valid = df_valid
    )

    if not is_ok:
        st.error(err_msg)
        if mismatches:
            for field, detail in mismatches.items():
                st.markdown(
                    f"- **{field}**: input `{detail['input']}` seharusnya `{detail['seharusnya']}`"
                )

        return

    
    x_dict = {
        "Month": tanggal.month,
        "DayOfWeek": tanggal.weekday(),
        "Gold_Price": float(harga_emas),
        "USD_Buy_Rate": float(usd_buy),
        "USD_Sell_Rate": float(usd_sell),
        "BI_Rate": float(bi_rate),
    }
    X_single = pd.DataFrame([x_dict])

    
    if not META_PATH.exists():
        st.error(f"Metadata model tidak ditemukan: {META_PATH}. Pastikan folder 'models/' berisi file .pkl dan 'model_meta.json'.")
        return

    # Prediksi Δ1..Δh dan rekonstruksi level
    levels, deltas = [], []
    prev_level = float(harga_emas)

    meta = load_meta(META_PATH)
    for h in range(1, horizon + 1):
        try:
            model, feat_cols, target_kind = load_model_for_horizon(h)
        except Exception as e:
            st.error(f"Model Δ{h} tidak bisa dimuat. Pastikan file ada di folder 'models/'.\nDetail: {e}")
            return

        # model single-shot harus meminta 6 fitur ini
        missing = [c for c in feat_cols if c not in X_single.columns]
        if missing:
            st.error(
                f"Model Δ{h} memerlukan fitur {missing}. "
            )
            return

        X_use = X_single.reindex(columns=feat_cols)
        pred = float(model.predict(X_use)[0])
        next_level, d = reconstruct_next(prev_level, pred, target_kind)
        levels.append(next_level)
        deltas.append(d)
        prev_level = next_level

    # Hasil
    dates = pd.date_range(start=tanggal + timedelta(days=1), periods=horizon, freq="D")
    out_df = pd.DataFrame({"Date": dates, "Pred_Level": levels, "Delta": deltas}).set_index("Date")

    st.subheader("Prediksi Harga")
    plot_df = out_df.copy()
    if df_valid is not None:
        actual_slice = (
            df_valid[(df_valid["Date"] >= plot_df.index.min()) &
                     (df_valid["Date"] <= plot_df.index.max())
                     ][["Date","Gold_Price"]]
                     .set_index("Date")
                     .sort_index()
                     
        )
        plot_df = plot_df.join(actual_slice, how="left")

    cols_to_plot = ["Pred_Level"]
    if "Gold_Price" in plot_df.columns:
        cols_to_plot.append("Gold_Price")

    
    rename_map = {
        "Pred_Level": "Harga Prediksi",
        "Gold_Price": "Harga Aktual"
    }

    plot_df_renamed = plot_df[cols_to_plot].rename(columns=rename_map)
    fig = go.Figure()

    # Garis Harga Prediksi 
    if "Harga Prediksi" in plot_df_renamed.columns:
        fig.add_trace(go.Scatter(
            x=plot_df_renamed.index,
            y=plot_df_renamed["Harga Prediksi"],
            mode="lines+markers",
            name="Harga Prediksi",
            line=dict(color="#FF991C", width=3),
            marker=dict(size=5)
        ))

    # Garis Harga Aktual 
    if "Harga Aktual" in plot_df_renamed.columns:
        fig.add_trace(go.Scatter(
            x=plot_df_renamed.index,
            y=plot_df_renamed["Harga Aktual"],
            mode="lines+markers",
            name="Harga Aktual",
            line=dict(color="#cca9dd", width=3, dash="dot"),
            marker=dict(size=5)
        ))

    fig.update_layout(
        height=380,
        xaxis_title="Tanggal",
        yaxis_title="Harga Emas (IDR/gram)",
        legend=dict(title=None, orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        template="simple_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Perubahan Harian (Harga prediksi)")
    st.bar_chart(out_df["Delta"])

    st.divider()
    st.subheader("Metrik Evaluasi")
    if METRICS_CSV.exists():
        try:
            mdf = pd.read_csv(METRICS_CSV)
            # tampilkan hanya horizon yang dipilih user
            row = mdf[mdf["Horizon"] == horizon]
            if not row.empty:
                hrow = row.iloc[0]
                st.metric("MAE (Level)", f"{hrow['MAE_level']:.2f}")
                st.metric("RMSE (Level)", f"{hrow['RMSE_level']:.2f}")
                st.metric("R² (Level)", f"{hrow['R2_level']:.4f}")
            else:
                st.info(f"Tidak ada metrik untuk Δ{horizon}.")
            
        except Exception as e:
            st.info(f"Tidak bisa membaca {METRICS_CSV}: {e}")
    else:
        st.info("File `models/metrics_all_horizons.csv` belum ada. Simpan saat training untuk menampilkan metrik di sini.")

    # Feature Importance sesuai horizon yang dipilih
    st.divider()
    st.subheader(f"Feature Importance (Model Δ{horizon})")
    st.markdown(
        """Note: Feature importance menunjukkan seberapa besar pengaruh setiap variabel terhadap hasil prediksi. LightGBM menghitung importance berdasarkan frekuensi dan besarnya peningkatan “gain” (informasi atau pengurangan error) yang diperoleh ketika suatu fitur digunakan untuk melakukan pemisahan (split) pada pohon keputusan di dalam model."""
    )
    try:
        mh, fh, th = load_model_for_horizon(horizon)
        imp = mh.feature_importances_
        order = np.argsort(imp)[::-1]
        mask = imp[order] > (0.01 * imp.max())
        fi = pd.DataFrame({"Fitur": np.array(fh)[order][mask], "Importance": imp[order][mask]}).set_index("Fitur")
        st.bar_chart(fi["Importance"])
    except Exception as e:
        st.info(f"FI Δ{horizon} tidak tersedia: {e}")


    if not submitted:
        return


    if DATASET_START <= tanggal <= DATASET_END:
        st.error("Tanggal harus di luar periode dataset (pilih setelah 31 Desember 2024).")
        return


    x_dict = {
        "Month": tanggal.month,
        "DayOfWeek": tanggal.weekday(),
        "Gold_Price": float(harga_emas),
        "USD_Buy_Rate": float(usd_buy),
        "USD_Sell_Rate": float(usd_sell),
        "BI_Rate": float(bi_rate),
    }
    X_single = pd.DataFrame([x_dict])

    # ----- Prediksi Δ1..Δh & rekonstruksi level -----
    levels, deltas = [], []
    prev_level = float(harga_emas)

    try:
        meta = load_meta(META_PATH)  
    except Exception as e:
        st.error(f"Metadata model tidak ditemukan ({META_PATH}). Pastikan folder 'models/' berisi file .pkl dan 'model_meta.json'.\nDetail: {e}")
        return

    for h in range(1, horizon + 1):
        try:
            model, feat_cols, target_kind = load_model_for_horizon(h)
        except Exception as e:
            st.error(f"Model Δ{h} tidak bisa dimuat. Pastikan file ada di folder 'models/'.\nDetail: {e}")
            return

        
        missing = [c for c in feat_cols if c not in X_single.columns]
        if missing:
            st.error(f"Model Δ{h} memerlukan fitur {missing}. Model kamu kemungkinan dilatih 'history-aware'. "
                     "Gunakan model single-shot (fitur: Month, DayOfWeek, Gold_Price, USD_Buy_Rate, USD_Sell_Rate, BI_Rate).")
            return

        X_use = X_single.reindex(columns=feat_cols)
        pred = float(model.predict(X_use)[0])
        next_level, d = reconstruct_next(prev_level, pred, target_kind)
        levels.append(next_level)
        deltas.append(d)
        prev_level = next_level

    if not submitted:
        return


    if DATASET_START <= tanggal <= DATASET_END:
        st.error("Tanggal harus di luar periode dataset (pilih setelah 31 Desember 2024).")
        return


    x_dict = {
        "Month": tanggal.month,
        "DayOfWeek": tanggal.weekday(),
        "Gold_Price": float(harga_emas),
        "USD_Buy_Rate": float(usd_buy),
        "USD_Sell_Rate": float(usd_sell),
        "BI_Rate": float(bi_rate),
    }
    X_single = pd.DataFrame([x_dict])

    # ----- Prediksi Δ1..Δh & rekonstruksi level -----
    levels, deltas = [], []
    prev_level = float(harga_emas)

    try:
        meta = load_meta(META_PATH)  
    except Exception as e:
        st.error(f"Metadata model tidak ditemukan ({META_PATH}). Pastikan folder 'models/' berisi file .pkl dan 'model_meta.json'.\nDetail: {e}")
        return

    for h in range(1, horizon + 1):
        try:
            model, feat_cols, target_kind = load_model_for_horizon(h)
        except Exception as e:
            st.error(f"Model Δ{h} tidak bisa dimuat. Pastikan file ada di folder 'models/'.\nDetail: {e}")
            return

        # align kolom (harus 6 fitur single-shot)
        missing = [c for c in feat_cols if c not in X_single.columns]
        if missing:
            st.error(f"Model Δ{h} memerlukan fitur {missing}. Model kamu kemungkinan dilatih 'history-aware'. "
                     "Gunakan model single-shot (fitur: Month, DayOfWeek, Gold_Price, USD_Buy_Rate, USD_Sell_Rate, BI_Rate).")
            return

        X_use = X_single.reindex(columns=feat_cols)
        pred = float(model.predict(X_use)[0])
        next_level, d = reconstruct_next(prev_level, pred, target_kind)
        levels.append(next_level)
        deltas.append(d)
        prev_level = next_level

    if not META_PATH.exists():
        st.error(f"Metadata model tidak ditemukan: {META_PATH}. Pastikan folder 'models/' berisi file .pkl dan 'model_meta.json'.")
        return



# Halaman: About
def render_about():
    st.title("About")
    st.write(
        lorem_ipsum :=
        "Aplikasi ini dirancang untuk memprediksi harga emas di Indonesia menggunakan model **LightGBM**.\n"
        
    )

    st.markdown("""
    <div style="text-align: left; margin-top: 20px; margin-left: 10px;">
        <a href="https://www.instagram.com/novandry_a" target="_blank" style="text-decoration:none;">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174855.png" 
                 width="22" style="vertical-align:middle; margin-right:6px;">
            <span style="font-size:16px; color:#E1306C;"><b>@novandry_a</b></span>
        </a>
        <br><br>
        <a href="mailto:novandryaprilian@gmail.com" style="text-decoration:none;">
            <img src="https://cdn-icons-png.flaticon.com/512/732/732200.png"
                 width="22" style="vertical-align:middle; margin-right:6px;">
            <span style="font-size:15px; color:#1E90FF;">novandryaprilian@gmail.com</span>
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("2025 Novandry Aprilian")

# Router Halaman
if nav == "Home":
    render_home()
elif nav == "Prediction":
    render_prediction()
else:
    render_about()

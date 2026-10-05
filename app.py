"""
College Feedback Sentiment Analysis
-----------------------------------
Streamlit dashboard: NLP + ML (TF-IDF + Multinomial Naive Bayes)

Run:
    pip install streamlit pandas numpy scikit-learn plotly
    streamlit run app.py

Expected CSV (placed next to app.py): columns
    feedback_id, category, feedback, sentiment
"""

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="College Feedback AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
CSV_CANDIDATES = [
    "college_feedback.csv",
    "feedback.csv",
    "student_feedback.csv",
    "college_feedback_sentiment.csv",
    "dataset.csv",
    "data.csv",
]
REQUIRED_COLS = {"feedback_id", "category", "feedback", "sentiment"}

NAVY = "#0B1F3A"
ROYAL = "#1D4ED8"
GREEN = "#16A34A"
RED = "#DC2626"
SLATE = "#64748B"

PAGES = [
    "📊  Dashboard",
    "🧠  Analyze Feedback",
    "💬  Student Feedback",
    "📈  Sentiment Analytics",
    "🎯  Model Performance",
    "📘  About Project",
]

# Keep negations - they carry sentiment
NEGATIONS = {"not", "no", "never", "nor", "none", "cannot", "n't", "nothing", "neither"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATIONS

# ----------------------------------------------------------------------------
# STYLING
# ----------------------------------------------------------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: #F1F5F9; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1350px; }}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {{ background: {NAVY}; border-right: none; }}
section[data-testid="stSidebar"] * {{ color: #E2E8F0; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}
.brand {{ display:flex; align-items:center; gap:12px; padding: 4px 4px 18px 4px;
          border-bottom: 1px solid rgba(255,255,255,0.10); margin-bottom: 18px; }}
.brand-logo {{ width:46px; height:46px; border-radius:12px; background:{ROYAL};
               display:flex; align-items:center; justify-content:center; font-size:24px; }}
.brand-title {{ font-weight:700; font-size:1.05rem; color:#FFFFFF !important; line-height:1.2; }}
.brand-sub {{ font-size:0.74rem; color:#94A3B8 !important; }}

section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 4px; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    padding: 11px 14px; border-radius: 10px; width: 100%; cursor: pointer;
    transition: background .15s ease; margin: 0;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: rgba(255,255,255,0.07); }}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{ display: none; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {{ font-size: 0.93rem; font-weight: 500; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
    background: {ROYAL}; box-shadow: 0 4px 14px rgba(29,78,216,0.45);
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{ color:#fff !important; font-weight:600; }}

.side-info {{ background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08);
              border-radius: 12px; padding: 14px 16px; margin-top: 28px; }}
.side-info .lbl {{ font-size: 0.68rem; letter-spacing: .08em; text-transform: uppercase; color:#94A3B8 !important; }}
.side-info .val {{ font-size: 0.88rem; font-weight: 600; color:#fff !important; margin-bottom: 10px; }}
.side-info .val:last-child {{ margin-bottom: 0; }}

/* ---------- Header ---------- */
.top-header {{ display:flex; justify-content:space-between; align-items:center;
               background:#fff; padding:20px 26px; border-radius:16px;
               box-shadow: 0 2px 10px rgba(15,23,42,0.06); margin-bottom: 22px;
               border-left: 5px solid {ROYAL}; }}
.top-header h1 {{ margin:0; font-size:1.55rem; font-weight:800; color:{NAVY}; }}
.top-header p {{ margin:2px 0 0 0; color:{SLATE}; font-size:0.9rem; }}
.profile {{ display:flex; align-items:center; gap:10px; }}
.profile .avatar {{ width:42px; height:42px; border-radius:50%; background:{NAVY}; color:#fff;
                    display:flex; align-items:center; justify-content:center; font-size:20px; }}
.profile .who {{ font-size:0.82rem; color:{NAVY}; font-weight:600; line-height:1.2; }}
.profile .role {{ font-size:0.72rem; color:{SLATE}; font-weight:400; }}

/* ---------- Cards ---------- */
.card {{ background:#fff; border-radius:16px; padding:22px 24px;
         box-shadow: 0 2px 10px rgba(15,23,42,0.06); margin-bottom: 20px; }}
.section-title {{ font-size:1.12rem; font-weight:700; color:{NAVY}; margin: 6px 0 2px 0; }}
.section-sub {{ font-size:0.84rem; color:{SLATE}; margin-bottom: 14px; }}

.kpi {{ background:#fff; border-radius:16px; padding:20px 22px; position:relative; overflow:hidden;
        box-shadow: 0 2px 10px rgba(15,23,42,0.06); }}
.kpi::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:5px; background: var(--c); }}
.kpi .icon {{ width:44px; height:44px; border-radius:12px; display:flex; align-items:center;
              justify-content:center; font-size:21px; background: var(--bg); float:right; }}
.kpi .label {{ font-size:0.8rem; font-weight:600; color:{SLATE}; text-transform:uppercase; letter-spacing:.05em; }}
.kpi .value {{ font-size:2.1rem; font-weight:800; color:{NAVY}; margin-top:6px; line-height:1.1; }}
.kpi .hint {{ font-size:0.78rem; color: var(--c); font-weight:600; margin-top:6px; }}

.pct-box {{ border-radius:14px; padding:18px 20px; margin-bottom:14px; }}
.pct-box .n {{ font-size:2.3rem; font-weight:800; line-height:1; }}
.pct-box .t {{ font-size:0.82rem; font-weight:600; margin-top:6px; }}
.pct-pos {{ background:#F0FDF4; color:{GREEN}; border:1px solid #BBF7D0; }}
.pct-neg {{ background:#FEF2F2; color:{RED}; border:1px solid #FECACA; }}

/* ---------- Badges / tables ---------- */
.badge {{ display:inline-block; padding:4px 12px; border-radius:999px; font-size:0.76rem; font-weight:700; }}
.badge-pos {{ background:#DCFCE7; color:#15803D; }}
.badge-neg {{ background:#FEE2E2; color:#B91C1C; }}
.cat-pill {{ display:inline-block; padding:3px 10px; border-radius:8px; background:#EFF6FF;
             color:{ROYAL}; font-size:0.78rem; font-weight:600; }}
table.fb {{ width:100%; border-collapse:collapse; }}
table.fb th {{ text-align:left; font-size:0.74rem; text-transform:uppercase; letter-spacing:.06em;
               color:{SLATE}; padding:10px 12px; border-bottom:2px solid #E2E8F0; background:#F8FAFC; }}
table.fb td {{ padding:12px; border-bottom:1px solid #F1F5F9; font-size:0.9rem; color:#1E293B; vertical-align:middle; }}
table.fb tr:hover td {{ background:#F8FAFC; }}
table.fb td.id {{ color:{SLATE}; font-weight:600; width:70px; }}

/* ---------- Result card ---------- */
.result {{ border-radius:18px; padding:28px; text-align:center; margin: 8px 0 18px 0; }}
.result.pos {{ background: linear-gradient(135deg,#F0FDF4,#DCFCE7); border:2px solid #86EFAC; }}
.result.neg {{ background: linear-gradient(135deg,#FEF2F2,#FEE2E2); border:2px solid #FCA5A5; }}
.result .head {{ font-size:1.7rem; font-weight:800; letter-spacing:.02em; }}
.result.pos .head {{ color:#15803D; }}
.result.neg .head {{ color:#B91C1C; }}
.result .conf {{ font-size:1.05rem; font-weight:600; color:{NAVY}; margin-top:6px; }}
.textbox {{ background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:14px 16px;
            color:#1E293B; font-size:0.92rem; line-height:1.55; }}
.textbox-label {{ font-size:0.74rem; font-weight:700; color:{SLATE}; text-transform:uppercase;
                  letter-spacing:.06em; margin-bottom:6px; }}

/* ---------- Workflow ---------- */
.flow-step {{ background:{NAVY}; color:#fff; border-radius:12px; padding:13px 18px; text-align:center;
              font-weight:600; font-size:0.92rem; max-width:360px; margin:0 auto; }}
.flow-step.end {{ background:{ROYAL}; }}
.flow-arrow {{ text-align:center; color:{ROYAL}; font-size:1.3rem; line-height:1.2; margin:3px 0; }}
.chip {{ display:inline-block; background:#EFF6FF; color:{ROYAL}; border:1px solid #BFDBFE;
         padding:7px 16px; border-radius:999px; font-weight:600; font-size:0.86rem; margin:4px 6px 4px 0; }}

/* ---------- Widgets ---------- */
.stTextArea textarea {{ border-radius:12px; border:1.5px solid #CBD5E1; font-size:0.97rem; padding:14px; background:#fff; }}
.stTextArea textarea:focus {{ border-color:{ROYAL}; box-shadow:0 0 0 3px rgba(29,78,216,.15); }}
.stButton > button {{ background:{ROYAL}; color:#fff; border:none; border-radius:10px; padding:0.65rem 1.6rem;
                      font-weight:600; font-size:0.95rem; box-shadow:0 4px 12px rgba(29,78,216,.30); }}
.stButton > button:hover {{ background:#1E40AF; color:#fff; }}
.stButton > button:focus {{ color:#fff; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# DATA + MODEL
# ----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, strip URLs/punctuation/digits, remove stop words (keep negations)."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.replace("n't", " not")
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def find_csv():
    for name in CSV_CANDIDATES:
        p = BASE_DIR / name
        if p.exists():
            return p
    for p in sorted(BASE_DIR.glob("*.csv")):
        try:
            cols = set(pd.read_csv(p, nrows=1).columns.str.strip().str.lower())
            if REQUIRED_COLS.issubset(cols):
                return p
        except Exception:
            continue
    return None


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing column(s): {', '.join(sorted(missing))}")
    df = df[list(REQUIRED_COLS)].dropna(subset=["feedback", "sentiment"])
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.capitalize()
    df = df[df["sentiment"].isin(["Positive", "Negative"])]
    df["category"] = df["category"].astype(str).str.strip().str.title()
    df["cleaned"] = df["feedback"].apply(clean_text)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_csv(path_str: str) -> pd.DataFrame:
    return prepare_df(pd.read_csv(path_str))


@st.cache_resource(show_spinner="Training sentiment model...")
def train_model(texts: tuple, labels: tuple):
    """TF-IDF + Multinomial Naive Bayes, 80/20 stratified split."""
    X_text = pd.Series(texts)
    y = pd.Series(labels)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_text, y, test_size=0.20, random_state=42, stratify=y
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    Xtr = vec.fit_transform(X_tr)
    Xte = vec.transform(X_te)
    clf = MultinomialNB(alpha=0.5)
    clf.fit(Xtr, y_tr)
    pred = clf.predict(Xte)
    labs = ["Positive", "Negative"]
    metrics = {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, pos_label="Positive", zero_division=0),
        "recall": recall_score(y_te, pred, pos_label="Positive", zero_division=0),
        "f1": f1_score(y_te, pred, pos_label="Positive", zero_division=0),
        "cm": confusion_matrix(y_te, pred, labels=labs),
        "labels": labs,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "n_features": len(vec.vocabulary_),
    }
    return vec, clf, metrics


def predict(vec, clf, text: str):
    cleaned = clean_text(text)
    if not cleaned:
        return None, None, cleaned
    X = vec.transform([cleaned])
    proba = clf.predict_proba(X)[0]
    idx = int(np.argmax(proba))
    return clf.classes_[idx], float(proba[idx]), cleaned


# ----------------------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------------------
def style_fig(fig, height=340):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#334155", size=12),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False)
    return fig


def header(title="College Feedback Analytics",
           sub="Student opinion monitoring and sentiment analysis system"):
    st.markdown(
        f"""
        <div class="top-header">
          <div><h1>{title}</h1><p>{sub}</p></div>
          <div class="profile">
            <div style="text-align:right"><div class="who">Admin</div><div class="role">Ayush Kumar</div></div>
            <div class="avatar">👤</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi(label, value, icon, color, bg, hint=""):
    return f"""
    <div class="kpi" style="--c:{color};--bg:{bg}">
      <div class="icon">{icon}</div>
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      <div class="hint">{hint}</div>
    </div>"""


def badge(sentiment):
    cls = "badge-pos" if sentiment == "Positive" else "badge-neg"
    dot = "●"
    return f'<span class="badge {cls}">{dot} {sentiment}</span>'


def feedback_table(df):
    rows = ""
    for _, r in df.iterrows():
        rows += (
            f"<tr><td class='id'>#{html.escape(str(r['feedback_id']))}</td>"
            f"<td><span class='cat-pill'>{html.escape(str(r['category']))}</span></td>"
            f"<td>{html.escape(str(r['feedback']))}</td>"
            f"<td>{badge(r['sentiment'])}</td></tr>"
        )
    return (
        "<table class='fb'><thead><tr><th>ID</th><th>Category</th><th>Feedback</th>"
        f"<th>Sentiment</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def section(title, sub=""):
    st.markdown(f"<div class='section-title'>{title}</div><div class='section-sub'>{sub}</div>",
                unsafe_allow_html=True)


def category_stats(df):
    g = df.groupby(["category", "sentiment"]).size().unstack(fill_value=0)
    for c in ("Positive", "Negative"):
        if c not in g.columns:
            g[c] = 0
    g["Total"] = g["Positive"] + g["Negative"]
    g["Positive %"] = (g["Positive"] / g["Total"] * 100).round(1)
    g["Negative %"] = (g["Negative"] / g["Total"] * 100).round(1)
    return g.reset_index()[["category", "Total", "Positive", "Negative", "Positive %", "Negative %"]]


COLOR_MAP = {"Positive": GREEN, "Negative": RED}


def donut(df, height=340):
    counts = df["sentiment"].value_counts().reindex(["Positive", "Negative"]).fillna(0)
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.62,
        marker=dict(colors=[GREEN, RED], line=dict(color="#fff", width=3)),
        textinfo="percent", textfont=dict(size=14, color="#fff"),
    ))
    fig.add_annotation(text=f"<b>{int(counts.sum())}</b><br>Total", showarrow=False,
                       font=dict(size=18, color=NAVY))
    return style_fig(fig, height)


def sentiment_bar(df, height=340):
    counts = df["sentiment"].value_counts().reindex(["Positive", "Negative"]).fillna(0).reset_index()
    counts.columns = ["Sentiment", "Count"]
    fig = px.bar(counts, x="Sentiment", y="Count", color="Sentiment", color_discrete_map=COLOR_MAP, text="Count")
    fig.update_traces(textposition="outside", marker_cornerradius=8, width=0.5)
    fig.update_layout(showlegend=False)
    return style_fig(fig, height)


def category_total_bar(df, height=360):
    c = df["category"].value_counts().reset_index()
    c.columns = ["Category", "Count"]
    fig = px.bar(c, x="Category", y="Count", text="Count")
    fig.update_traces(marker_color=ROYAL, marker_cornerradius=8, textposition="outside")
    return style_fig(fig, height)


def category_sentiment_bar(df, height=360):
    c = df.groupby(["category", "sentiment"]).size().reset_index(name="Count")
    fig = px.bar(c, x="category", y="Count", color="sentiment", barmode="group",
                 color_discrete_map=COLOR_MAP, labels={"category": "Category", "sentiment": "Sentiment"})
    fig.update_traces(marker_cornerradius=6)
    return style_fig(fig, height)


# ----------------------------------------------------------------------------
# PAGES
# ----------------------------------------------------------------------------
def page_dashboard(df, metrics):
    header()
    total = len(df)
    pos = int((df["sentiment"] == "Positive").sum())
    neg = total - pos
    pos_pct = pos / total * 100 if total else 0
    neg_pct = neg / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Total Feedback", f"{total}", "📝", ROYAL, "#EFF6FF", "Records in dataset"), unsafe_allow_html=True)
    c2.markdown(kpi("Positive Feedback", f"{pos}", "😊", GREEN, "#F0FDF4", f"▲ {pos_pct:.1f}% of total"), unsafe_allow_html=True)
    c3.markdown(kpi("Negative Feedback", f"{neg}", "⚠️", RED, "#FEF2F2", f"▼ {neg_pct:.1f}% of total"), unsafe_allow_html=True)
    c4.markdown(kpi("Model Accuracy", f"{metrics['accuracy']*100:.1f}%", "🎯", NAVY, "#E2E8F0", "On 20% test split"), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Sentiment overview
    with st.container(border=True):
        section("Student Sentiment Overview", "Overall distribution of positive and negative student opinions")
        a, b, c = st.columns([1, 1.5, 1.5])
        with a:
            st.markdown(
                f"<div class='pct-box pct-pos'><div class='n'>{pos_pct:.1f}%</div><div class='t'>😊 POSITIVE</div></div>"
                f"<div class='pct-box pct-neg'><div class='n'>{neg_pct:.1f}%</div><div class='t'>⚠️ NEGATIVE</div></div>",
                unsafe_allow_html=True,
            )
        with b:
            st.plotly_chart(donut(df), use_container_width=True, config={"displayModeBar": False})
        with c:
            st.plotly_chart(sentiment_bar(df), use_container_width=True, config={"displayModeBar": False})

    # Category analysis
    with st.container(border=True):
        section("Feedback by Category", "Number of feedback records and sentiment comparison for each campus area")
        l, r = st.columns(2)
        with l:
            st.markdown("**Total feedback per category**")
            st.plotly_chart(category_total_bar(df), use_container_width=True, config={"displayModeBar": False})
        with r:
            st.markdown("**Sentiment comparison by category**")
            st.plotly_chart(category_sentiment_bar(df), use_container_width=True, config={"displayModeBar": False})

    # Recent feedback
    with st.container(border=True):
        section("Recent Student Feedback", "Latest 10 feedback entries")
        recent = df.tail(10).iloc[::-1]
        st.markdown(feedback_table(recent), unsafe_allow_html=True)


SAMPLE_FEEDBACK = [
    "The faculty are very supportive and explain every concept clearly.",
    "The hostel WiFi is extremely slow and keeps disconnecting all day.",
    "The library has a great collection of books and a peaceful reading environment.",
    "Canteen food is overpriced and the hygiene is not good at all.",
    "The placement cell trains students well and many companies visit our campus.",
    "Laboratory equipment is outdated and often not working during practicals.",
    "Campus is clean, green and has excellent facilities for students.",
    "The administration office is slow and staff do not respond to our queries.",
    "Classrooms are spacious, well ventilated and equipped with projectors.",
    "Hostel rooms are overcrowded and the water supply is irregular.",
]


def next_sample():
    """Cycle to the next sample feedback every time the button is clicked."""
    i = st.session_state.get("sample_idx", -1) + 1
    st.session_state["sample_idx"] = i % len(SAMPLE_FEEDBACK)
    st.session_state["fb_input"] = SAMPLE_FEEDBACK[st.session_state["sample_idx"]]


def page_analyze(vec, clf):
    header("Analyze Feedback", "Predict the sentiment of any student feedback using the trained model")
    with st.container(border=True):
        section("Enter Feedback", "Type or paste a student's comment and run the classifier")
        text = st.text_area("Feedback", height=160, placeholder="Enter student feedback...",
                            label_visibility="collapsed", key="fb_input")
        b1, b2, _ = st.columns([1.1, 1, 4])
        analyze = b1.button("Analyze Sentiment", type="primary")
        b2.button("Try sample", on_click=next_sample)

    if analyze:
        if not text.strip():
            st.warning("Please enter some feedback text before analyzing.")
            return
        label, conf, cleaned = predict(vec, clf, text)
        if label is None:
            st.warning("The feedback has no meaningful words after cleaning. Try a longer sentence.")
            return
        is_pos = label == "Positive"
        st.markdown(
            f"""<div class="result {'pos' if is_pos else 'neg'}">
                  <div class="head">{'😊 POSITIVE FEEDBACK' if is_pos else '⚠️ NEGATIVE FEEDBACK'}</div>
                  <div class="conf">Confidence: {conf*100:.0f}%</div>
                </div>""",
            unsafe_allow_html=True,
        )
        st.progress(min(max(conf, 0.0), 1.0))
        l, r = st.columns(2)
        l.markdown(f"<div class='textbox-label'>Original Feedback</div><div class='textbox'>{html.escape(text)}</div>",
                   unsafe_allow_html=True)
        r.markdown(f"<div class='textbox-label'>Cleaned Feedback</div><div class='textbox'>{html.escape(cleaned)}</div>",
                   unsafe_allow_html=True)

        probs = dict(zip(clf.classes_, clf.predict_proba(vec.transform([cleaned]))[0]))
        pf = pd.DataFrame({"Sentiment": list(probs.keys()), "Probability": [v * 100 for v in probs.values()]})
        fig = px.bar(pf, x="Probability", y="Sentiment", orientation="h", color="Sentiment",
                     color_discrete_map=COLOR_MAP, text=pf["Probability"].map(lambda v: f"{v:.1f}%"))
        fig.update_traces(marker_cornerradius=8)
        fig.update_layout(showlegend=False, xaxis_range=[0, 100])
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            section("Class Probabilities", "How the model scored each sentiment")
            st.plotly_chart(style_fig(fig, 200), use_container_width=True, config={"displayModeBar": False})


def page_feedback(df):
    header("Student Feedback", "Browse and filter every feedback record in the dataset")
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.2, 1.2, 2])
        cats = ["All"] + sorted(df["category"].unique())
        cat = c1.selectbox("Category", cats)
        sen = c2.selectbox("Sentiment", ["All", "Positive", "Negative"])
        q = c3.text_input("Search", placeholder="Search keywords e.g. wifi, library, teaching...")
        f = df
        if cat != "All":
            f = f[f["category"] == cat]
        if sen != "All":
            f = f[f["sentiment"] == sen]
        if q.strip():
            f = f[f["feedback"].str.contains(q.strip(), case=False, na=False, regex=False)]
        st.caption(f"Showing {len(f)} of {len(df)} records")
        n = st.slider("Rows to display", 5, 100, 25, step=5) if len(f) > 5 else len(f)
        if len(f):
            st.markdown(feedback_table(f.head(n)), unsafe_allow_html=True)
        else:
            st.info("No feedback matches the selected filters.")


def page_analytics(df):
    header("Sentiment Analytics", "Detailed breakdown of student sentiment across categories")
    total = len(df)
    pos = int((df["sentiment"] == "Positive").sum())
    neg = total - pos
    stats = category_stats(df)
    best = stats.sort_values(["Positive %", "Total"], ascending=False).iloc[0]
    worst = stats.sort_values(["Negative %", "Total"], ascending=False).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Positive", f"{pos/total*100:.1f}%", "😊", GREEN, "#F0FDF4", f"{pos} feedback"), unsafe_allow_html=True)
    c2.markdown(kpi("Negative", f"{neg/total*100:.1f}%", "⚠️", RED, "#FEF2F2", f"{neg} feedback"), unsafe_allow_html=True)
    c3.markdown(kpi("Most Positive Category", best["category"], "🏆", GREEN, "#F0FDF4", f"{best['Positive %']:.1f}% positive"), unsafe_allow_html=True)
    c4.markdown(kpi("Most Negative Category", worst["category"], "📉", RED, "#FEF2F2", f"{worst['Negative %']:.1f}% negative"), unsafe_allow_html=True)
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    l, r = st.columns([1, 1.6])
    with l:
        with st.container(border=True):
            section("Sentiment Distribution", "Share of each sentiment")
            st.plotly_chart(donut(df, 330), use_container_width=True, config={"displayModeBar": False})
    with r:
        with st.container(border=True):
            section("Sentiment by Category", "Positive vs negative feedback in each category")
            st.plotly_chart(category_sentiment_bar(df, 330), use_container_width=True, config={"displayModeBar": False})

    with st.container(border=True):
        section("Positive Share by Category", "Percentage of positive feedback (higher is better)")
        s = stats.sort_values("Positive %")
        fig = px.bar(s, x="Positive %", y="category", orientation="h", text="Positive %",
                     labels={"category": ""}, color="Positive %",
                     color_continuous_scale=[(0, "#FCA5A5"), (0.5, "#FDE68A"), (1, "#4ADE80")])
        fig.update_traces(marker_cornerradius=6, texttemplate="%{text}%")
        fig.update_layout(coloraxis_showscale=False, xaxis_range=[0, 105])
        st.plotly_chart(style_fig(fig, 360), use_container_width=True, config={"displayModeBar": False})

    with st.container(border=True):
        section("Category Summary", "Counts and percentages per category")
        st.dataframe(
            stats.rename(columns={"category": "Category"}),
            hide_index=True, use_container_width=True,
            column_config={
                "Positive %": st.column_config.ProgressColumn("Positive %", min_value=0, max_value=100, format="%.1f%%"),
                "Negative %": st.column_config.ProgressColumn("Negative %", min_value=0, max_value=100, format="%.1f%%"),
            },
        )


def page_model(metrics, df):
    header("Model Performance", "Evaluation of the sentiment classifier on the held-out test set")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Accuracy", f"{metrics['accuracy']*100:.1f}%", "🎯", ROYAL, "#EFF6FF", "Overall correctness"), unsafe_allow_html=True)
    c2.markdown(kpi("Precision", f"{metrics['precision']*100:.1f}%", "🔍", GREEN, "#F0FDF4", "Positive class"), unsafe_allow_html=True)
    c3.markdown(kpi("Recall", f"{metrics['recall']*100:.1f}%", "📡", "#7C3AED", "#F5F3FF", "Positive class"), unsafe_allow_html=True)
    c4.markdown(kpi("F1 Score", f"{metrics['f1']*100:.1f}%", "⚖️", NAVY, "#E2E8F0", "Harmonic mean"), unsafe_allow_html=True)
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    l, r = st.columns([1.2, 1])
    with l:
        with st.container(border=True):
            section("Confusion Matrix", "Rows = actual, columns = predicted")
            cm = metrics["cm"]
            labs = metrics["labels"]
            fig = px.imshow(cm, x=labs, y=labs, text_auto=True,
                            color_continuous_scale=[(0, "#EFF6FF"), (1, ROYAL)],
                            labels=dict(x="Predicted", y="Actual", color="Count"))
            fig.update_traces(textfont=dict(size=22))
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig, 380), use_container_width=True, config={"displayModeBar": False})
    with r:
        with st.container(border=True):
            section("Model Information", "Configuration used for training")
            info = {
                "Model": "Multinomial Naive Bayes",
                "Feature Extraction": "TF-IDF (unigrams + bigrams)",
                "Train / Test Split": "80% / 20% (stratified)",
                "Training Samples": metrics["n_train"],
                "Testing Samples": metrics["n_test"],
                "Vocabulary Size": f"{metrics['n_features']} features",
                "Classes": "Positive, Negative",
            }
            rows = "".join(
                f"<tr><td style='color:{SLATE};font-weight:600;width:48%'>{k}</td>"
                f"<td style='color:{NAVY};font-weight:600'>{v}</td></tr>" for k, v in info.items()
            )
            st.markdown(f"<table class='fb'><tbody>{rows}</tbody></table>", unsafe_allow_html=True)


def page_about():
    header("About Project", "Overview of the College Feedback Sentiment Analysis system")
    l, r = st.columns([1.1, 1])
    with l:
        with st.container(border=True):
            section("Project Title")
            st.markdown(f"<h3 style='color:{NAVY};margin:0 0 14px 0'>College Feedback Sentiment Analysis</h3>",
                        unsafe_allow_html=True)
            section("Objective")
            st.write("Analyze student feedback using NLP and Machine Learning to automatically identify "
                     "whether an opinion is positive or negative, helping the college monitor student "
                     "satisfaction across faculty, facilities and services.")
            section("Technologies")
            chips = ["Python", "Pandas", "NumPy", "Scikit-learn", "TF-IDF", "Naive Bayes", "Streamlit"]
            st.markdown("".join(f"<span class='chip'>{c}</span>" for c in chips), unsafe_allow_html=True)
    with r:
        with st.container(border=True):
            section("Workflow", "From raw feedback to summary")
            steps = ["Student Feedback", "Text Cleaning", "TF-IDF", "Train/Test Split",
                     "Sentiment Classifier", "Positive / Negative Prediction", "Feedback Summary"]
            out = ""
            for i, s in enumerate(steps):
                cls = "flow-step end" if i == len(steps) - 1 else "flow-step"
                out += f"<div class='{cls}'>{s}</div>"
                if i < len(steps) - 1:
                    out += "<div class='flow-arrow'>↓</div>"
            st.markdown(out, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def sidebar(n_records, model_label):
    with st.sidebar:
        st.markdown(
            """<div class="brand"><div class="brand-logo">🎓</div>
               <div><div class="brand-title">College Feedback AI</div>
               <div class="brand-sub">Student Sentiment Analytics</div></div></div>""",
            unsafe_allow_html=True,
        )
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.markdown(
            f"""<div class="side-info">
                  <div class="lbl">Dataset</div><div class="val">{n_records} Records</div>
                  <div class="lbl">Model</div><div class="val">{model_label}</div>
                </div>""",
            unsafe_allow_html=True,
        )
    return page


def load_data():
    """Return dataframe or None (after showing a friendly error)."""
    path = find_csv()
    if path is not None:
        try:
            return load_csv(str(path))
        except Exception as e:
            st.error(f"Could not read `{path.name}`: {e}")
            return None

    st.markdown(
        "<div class='card'><h3 style='color:#B91C1C;margin-top:0'>⚠️ Dataset not found</h3>"
        "Place your feedback CSV in the same folder as <code>app.py</code> "
        "(for example <code>college_feedback.csv</code>) with columns "
        "<code>feedback_id, category, feedback, sentiment</code>, or upload it below.</div>",
        unsafe_allow_html=True,
    )
    up = st.file_uploader("Upload feedback CSV", type="csv")
    if up is not None:
        try:
            return prepare_df(pd.read_csv(up))
        except Exception as e:
            st.error(f"Invalid CSV: {e}")
    return None


def main():
    df = load_data()
    if df is None:
        sidebar(0, "TF-IDF + Naive Bayes")
        st.stop()
    if df["sentiment"].nunique() < 2 or len(df) < 10:
        st.error("The dataset needs both Positive and Negative feedback (at least 10 rows) to train the model.")
        st.stop()

    vec, clf, metrics = train_model(tuple(df["cleaned"]), tuple(df["sentiment"]))
    page = sidebar(len(df), "TF-IDF + Naive Bayes")

    if page == PAGES[0]:
        page_dashboard(df, metrics)
    elif page == PAGES[1]:
        page_analyze(vec, clf)
    elif page == PAGES[2]:
        page_feedback(df)
    elif page == PAGES[3]:
        page_analytics(df)
    elif page == PAGES[4]:
        page_model(metrics, df)
    else:
        page_about()


main()
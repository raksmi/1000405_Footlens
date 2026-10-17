import warnings
warnings.filterwarnings("ignore")

import functools
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


try:
    import statsmodels  
    TREND = "ols"
except ImportError:
    TREND = None

st.set_page_config(page_title="FootLens | Match Intelligence", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")
px.defaults.color_discrete_sequence = ["#38E58C", "#FF8A65", "#8DEBC0", "#C6F7DE", "#F4D35E", "#A4C9B8"]
px.defaults.template = "plotly_dark"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --bg:#0b1510; --panel:#14251b; --panel2:#1b3325; --lime:#39e58c; --orange:#ff8a65; --cream:#f2fff7; --muted:#a8c5b3; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 15% 0%, #17452d 0%, #0b1510 45%, #08110c 100%); color:var(--cream); }
.block-container { padding-top:1.7rem; padding-bottom:3rem; max-width:1440px; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; color:var(--cream); }
h1 { font-size:2.55rem !important; font-weight:700 !important; }
h2 { font-size:1.55rem !important; }
p, label, .stCaption { color:var(--muted); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#163222 0%,#0d1912 100%); border-right:1px solid #334236; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:var(--lime); }
[data-testid="stMetric"] { background:linear-gradient(145deg,#1c3828,#122219); border:1px solid #344537; border-radius:16px; padding:16px 18px; box-shadow:0 8px 24px #00000020; }
[data-testid="stMetricLabel"] { color:#a9b5a8 !important; font-size:.8rem; }
[data-testid="stMetricValue"] { color:#f4f2e9 !important; font-family:'Space Grotesk',sans-serif; }
.stButton > button, .stDownloadButton > button { background:var(--lime); color:#172016; border:0; border-radius:10px; font-weight:700; transition:all .18s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { background:#8bffc0; color:#111; transform:translateY(-1px); }
.stSelectbox [data-baseweb="select"], .stMultiSelect [data-baseweb="select"], .stTextInput input, .stNumberInput input { background:#1b3325; border-color:#37684a; border-radius:10px; }
div[data-testid="stPlotlyChart"] { background:transparent; border:0; border-radius:0; padding:0; margin:18px 0 28px 0; }
[data-testid="stDataFrame"] { border:1px solid #344537; border-radius:12px; overflow:hidden; }
hr { border-color:#28543a; }
.fl-hero { background:linear-gradient(120deg,#164b30 0%,#153b27 58%,#19452d 100%); border:0; border-radius:22px; padding:30px 32px; margin:0 0 20px 0; }
.fl-kicker { color:var(--lime); text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }
.fl-title { font-family:'Space Grotesk',sans-serif; font-size:2.65rem; line-height:1.04; font-weight:700; color:#f4f2e9; margin:9px 0 12px; }
.fl-copy { color:#bac7b8; max-width:680px; font-size:1rem; }
.fl-mascot { min-height:180px; display:flex; align-items:center; justify-content:center; text-align:center; padding:4px; color:#39e58c; }
.fl-mascot-frame { background:transparent; border:0; border-radius:0; padding:0; overflow:visible; box-shadow:none; }
.fl-mascot-frame img { border-radius:0; box-shadow:none; }
.fl-insight { border-left:4px solid #39e58c; background:linear-gradient(100deg,#17452d,#14251b); border-radius:0 14px 14px 0; padding:16px 18px; margin:18px 0 8px; }
.fl-insight-title { color:#39e58c; text-transform:uppercase; font-size:.72rem; letter-spacing:.12em; font-weight:700; margin-bottom:6px; }
.fl-small { font-size:.85rem; color:#a9b5a8; }
button[data-baseweb="tab"] { color:#a9b5a8; }
button[data-baseweb="tab"][aria-selected="true"] { color:#39e58c; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]



@st.cache_data
def load_data():
    p = BASE_DIR / "player_injuries_impact.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def rating_num(s):
    return pd.to_numeric(s.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce")


def row_mean(data, cols):
    cols = [c for c in cols if c in data.columns]
    if not cols:
        return pd.Series(np.nan, index=data.index)
    return data[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def result_counts(data, prefix):
    cols = [f"Match{i}_{prefix}_Result" for i in range(1, 4) if f"Match{i}_{prefix}_Result" in data.columns]
    if not cols:
        return {"W": 0, "D": 0, "L": 0}
    vals = data[cols].stack().astype(str).str.upper().str.strip()
    vals = vals.replace({"WIN": "W", "DRAW": "D", "LOSS": "L", "LOSE": "L"})
    return {x: int((vals == x).sum()) for x in ["W", "D", "L"]}


def win_prob(data, prefix):
    c = result_counts(data, prefix)
    n = sum(c.values())
    return c["W"] / n if n else np.nan


def clean_plot(data, cols):
    z = data.copy()
    for c in cols:
        if c in z.columns:
            z[c] = pd.to_numeric(z[c], errors="coerce")
    return z.dropna(subset=[c for c in cols if c in z.columns])


def corr(data, x, y):
    z = clean_plot(data, [x, y])
    return z[x].corr(z[y]) if len(z) >= 3 else np.nan


def pct(x):
    return f"{x:.1%}" if pd.notna(x) else "n/a"


def prepare(df):
    """Clean the raw CSV and add the derived columns used across the app."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    # tidy text columns; treat blanks and the string "nan" as missing
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            s = df[c].astype(str).str.strip()
            df[c] = s.where(~s.str.lower().isin(["nan", "none", ""]), np.nan)

    for c in ["Age", "FIFA rating", "GD"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["Date of Injury", "Date of return"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in df.columns:
        if "Player_rating" in c:
            df[c] = rating_num(df[c])

    bg = [f"Match{i}_before_injury_GD" for i in range(1, 4)]
    dg = [f"Match{i}_missed_match_GD" for i in range(1, 4)]
    ag = [f"Match{i}_after_injury_GD" for i in range(1, 4)]
    br = [f"Match{i}_before_injury_Player_rating" for i in range(1, 4)]
    ar = [f"Match{i}_after_injury_Player_rating" for i in range(1, 4)]

    df["Injury Duration"] = (df["Date of return"] - df["Date of Injury"]).dt.days
    # a return date before the injury date is a data entry error, so treat it as missing
    df.loc[df["Injury Duration"] < 0, "Injury Duration"] = np.nan

    df["Avg GD Before"] = row_mean(df, bg)
    df["Avg GD During"] = row_mean(df, dg)
    df["Avg GD After"] = row_mean(df, ag)
    df["Team Performance Drop"] = df["Avg GD Before"] - df["Avg GD During"]
    df["Avg Rating Before"] = row_mean(df, br)
    df["Avg Rating After"] = row_mean(df, ar)
    df["Rating Change"] = df["Avg Rating After"] - df["Avg Rating Before"]
    df["Injury Month"] = df["Date of Injury"].dt.month_name()
    df["Month Number"] = df["Date of Injury"].dt.month
    df["Impact Level"] = df["Team Performance Drop"].apply(
        lambda x: "Unknown" if pd.isna(x) else "High" if x > .75 else "Moderate" if x > .25 else "Low")
    return df


def init():
    raw = load_data()
    if raw is None:
        st.error("player_injuries_impact.csv is missing. Put it in the same folder as app.py.")
        st.stop()
    return prepare(raw)


FILTERS = {
    "f_season": ("Season", "Season"),
    "f_club": ("Club", "Team Name"),
    "f_position": ("Position", "Position"),
    "f_player": ("Player", "Name"),
    "f_injury": ("Injury type", "Injury"),
    "f_impact": ("Impact level", "Impact Level"),
}


def reset_filters():
    for k in list(FILTERS) + ["f_age"]:
        st.session_state.pop(k, None)


def apply_filters(df, selections, age_range=None, age_bounds=None):
    out = df
    for col, chosen in selections.items():
        if chosen:
            out = out[out[col].astype(str).isin(chosen)]
    if age_range and age_bounds and tuple(age_range) != tuple(age_bounds):
        out = out[out["Age"].between(*age_range)]
    return out


def sidebar_filters(df):
    st.sidebar.title("FootLens")
    st.sidebar.caption("Player injuries and team performance")
    st.sidebar.subheader("Filters")
    st.sidebar.caption("Leave a filter empty to include everything.")

    selections = {}
    for key, (label, col) in FILTERS.items():
        options = ["High", "Moderate", "Low", "Unknown"] if col == "Impact Level" \
            else sorted(df[col].dropna().astype(str).unique().tolist())
        selections[col] = st.sidebar.multiselect(label, options, key=key, placeholder="All")

    age_range = age_bounds = None
    ages = df["Age"].dropna()
    if len(ages) and int(ages.min()) < int(ages.max()):
        age_bounds = (int(ages.min()), int(ages.max()))
        age_range = st.sidebar.slider("Age", age_bounds[0], age_bounds[1], age_bounds, key="f_age")

    out = apply_filters(df, selections, age_range, age_bounds)
    st.sidebar.button("Reset filters", on_click=reset_filters, use_container_width=True)
    st.sidebar.caption(f"Showing {len(out):,} of {len(df):,} records")
    return out


def active_filter_text():
    bits = []
    for key, (label, _) in FILTERS.items():
        chosen = st.session_state.get(key)
        if chosen:
            bits.append(f"{label}: {', '.join(map(str, chosen[:3]))}" + (f" +{len(chosen) - 3}" if len(chosen) > 3 else ""))
    age = st.session_state.get("f_age")
    if age and len(data) != len(df) and not bits:
        bits.append(f"Age {age[0]}-{age[1]}")
    return "  |  ".join(bits) if bits else "No filters applied"


def show(fig, height=420):
    palette = ["#39E58C", "#FF8A65", "#8DEBC0", "#C6F7DE", "#F4D35E", "#A4C9B8"]
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=62, b=20),
        legend_title_text="",
        paper_bgcolor="#14251B",
        plot_bgcolor="#14251B",
        font=dict(family="DM Sans, sans-serif", color="#E8FFF0", size=12),
        title=dict(font=dict(family="Space Grotesk, sans-serif", size=17, color="#F2FFF7"), x=0.02, xanchor="left"),
        colorway=palette,
        xaxis=dict(gridcolor="#2B382D", zerolinecolor="#425342", linecolor="#425342"),
        yaxis=dict(gridcolor="#2B382D", zerolinecolor="#425342", linecolor="#425342"),
    )
    
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True})

def insight_box(title, body):
    st.markdown(
        f'<div class="fl-insight"><div class="fl-insight-title">{title}</div><div>{body}</div></div>',
        unsafe_allow_html=True
    )


def header(title, question=None):
    st.markdown(f'<div class="fl-kicker">FOOTLENS ANALYTICS  /  MATCH INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title(title)
    if question:
        st.markdown(f'<p class="fl-copy">{question}</p>', unsafe_allow_html=True)


def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.metric(label, value, help=sub)


def page(fn):
    """Show a friendly message (instead of crashing) when the filters leave nothing."""
    @functools.wraps(fn)
    def wrapper():
        if data.empty:
            st.warning("No records match the current filters.")
            st.button("Reset filters", on_click=reset_filters, key=f"reset_{fn.__name__}")
            return
        fn()
    return wrapper


@page
def home():
   
    mascot_candidates = [
        BASE_DIR / name for name in (
            "mascot.png", "mascot.jpg", "mascot.jpeg", "mascot.webp",
            "footlens_mascot.png", "footlens_mascot.jpg", "FootLens_Mascot.png",
            "FootLens mascot.png", "logo_mascot.png"
        )
    ]
    mascot_path = next((p for p in mascot_candidates if p.exists()), None)
    if mascot_path is None:
        mascot_path = next((p for p in BASE_DIR.iterdir() if p.is_file() and "mascot" in p.stem.lower() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}), None)

    hero_left, hero_right = st.columns([1.65, 0.8], gap="large", vertical_alignment="center")
    with hero_left:
        st.markdown("""
        <div class="fl-hero">
          <div class="fl-kicker">THE INJURY INTELLIGENCE HUB</div>
          <div class="fl-title">Every absence<br>leaves a footprint.</div>
          <div class="fl-copy">Explore how player injuries connect with team results, recovery form and club-level performance — all in one match-analysis workspace.</div>
          <div style="margin-top:20px;color:#8fffc0;font-size:.78rem;font-weight:700;letter-spacing:.1em">DATA-DRIVEN · PLAYER-CENTRED · MATCH-READY</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(active_filter_text())
    with hero_right:
        if mascot_path:
            
            st.image(str(mascot_path), use_container_width=True)
        else:
            st.markdown("""
            <div class="fl-mascot"><div>
              <div style="font-size:3.2rem">⚽</div>
              <div style="font-weight:700;font-size:1.1rem">FOOTLENS</div>
            </div></div>
            """, unsafe_allow_html=True)

    dur = data["Injury Duration"]
    metric_row([
        ("Records", f"{len(data):,}", "Rows after filtering"),
        ("Players", f"{data['Name'].nunique():,}", "Unique players"),
        ("Clubs", f"{data['Team Name'].nunique():,}", "Unique clubs"),
        ("Avg injury length", f"{dur.mean():.0f} days" if dur.notna().any() else "n/a", "Return date minus injury date"),
        ("Avg goal diff. drop", f"{data['Team Performance Drop'].mean():.2f}", "Before injury minus while absent"),
        ("Win rate while absent", pct(win_prob(data, "missed_match")), "Across the 3 matches missed"),
    ])

    st.subheader("Quick summary")
    inj = data["Injury"].value_counts()
    club = data.groupby("Team Name")["Team Performance Drop"].mean().dropna()
    rc = data["Rating Change"].dropna()
    if len(inj):
        st.write(f"- The most common injury is **{inj.index[0]}** ({int(inj.iloc[0])} records).")
    if len(club):
        st.write(f"- **{club.idxmax()}** has the biggest average drop in goal difference ({club.max():.2f}).")
    if len(rc):
        st.write(f"- {(rc > 0).mean():.0%} of players with ratings on both sides of the injury improved after returning.")

    left, right = st.columns(2)
    with left:
        monthly = data.groupby("Month Number").size().reindex(range(1, 13), fill_value=0)
        md = pd.DataFrame({"Month": MONTHS, "Injuries": monthly.values})
        show(px.line(md, x="Month", y="Injuries", markers=True, title="Injuries by month"), 360)
    with right:
        rec = pd.DataFrame({"Period": ["Before injury", "While absent"],
                            "Win rate": [win_prob(data, "before_injury"), win_prob(data, "missed_match")]}).dropna()
        fig = px.bar(rec, x="Period", y="Win rate", range_y=[0, 1], text="Win rate", title="Win rate before vs. during absence")
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        show(fig, 360)

   
    if len(inj):
        insight_box("What the dashboard suggests", f"The most frequently recorded injury is <b>{inj.index[0]}</b> ({int(inj.iloc[0])} records). Frequency alone does not mean it has the greatest performance impact.")
    if pd.notna(win_prob(data, "before_injury")) and pd.notna(win_prob(data, "missed_match")):
        delta = (win_prob(data, "missed_match") - win_prob(data, "before_injury")) * 100
        insight_box("Win-rate signal", f"Observed win rate changed by <b>{delta:+.1f} percentage points</b> between the pre-injury and absence windows. Treat this as an association, not proof of causation.")
    st.caption("Pick a research question from the navigation at the top. Sidebar filters apply across the analysis pages.")


@page
def rq1():
    header("RQ1: Which injuries hurt the team most?",
           "Which types of player injuries are associated with the greatest decline in team performance?")
    q = (data.groupby("Injury")
         .agg(Cases=("Injury", "size"), Avg_Drop=("Team Performance Drop", "mean"), Avg_Duration=("Injury Duration", "mean"))
         .dropna(subset=["Avg_Drop"]).reset_index().sort_values("Avg_Drop", ascending=False))
    if q.empty:
        st.info("Not enough valid data for this view.")
        return
    top = q.iloc[0]
    metric_row([("Worst injury type", top["Injury"], "Highest average drop"),
                ("Avg drop", f"{top['Avg_Drop']:.2f}", "Goal difference"),
                ("Cases", str(int(top["Cases"])), "Records for this injury")])
    st.write(f"**{top['Injury']}** has the largest average drop in team goal difference, at {top['Avg_Drop']:.2f}.")
    left, right = st.columns(2)
    with left:
        fig = px.bar(q.head(12).sort_values("Avg_Drop"), x="Avg_Drop", y="Injury", orientation="h", text="Avg_Drop",
                     hover_data=["Cases", "Avg_Duration"], title="Average drop by injury type (top 12)")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        show(fig)
    with right:
        z = q.dropna(subset=["Cases", "Avg_Drop", "Avg_Duration"]).copy()
        z["Avg_Duration"] = z["Avg_Duration"].clip(lower=0)  # marker size can't be negative
        if len(z):
            show(px.scatter(z, x="Cases", y="Avg_Drop", size="Avg_Duration", hover_name="Injury",
                            title="How common vs. how damaging (bubble = avg days out)",
                            labels={"Cases": "Records", "Avg_Drop": "Average drop"}))
        else:
            st.info("No injury durations available for the bubble chart.")
    valid = data[["Injury", "Injury Duration"]].dropna()
    if not valid.empty:
        fig = px.box(valid, y="Injury Duration", x="Injury", points=False, title="Time out by injury type (days)")
        fig.update_xaxes(tickangle=-35)
        show(fig)


    insight_box("How to read these graphs", f"The highest average performance-drop category is <b>{top['Injury']}</b> ({top['Avg_Drop']:.2f} goal-difference points across {int(top['Cases'])} records). Check the case count and duration alongside the average: small groups can produce unstable rankings.")


@page
def rq2():
    header("RQ2: Results while a player is out",
           "How does a team's win-loss record change while a key player is absent?")
    b = result_counts(data, "before_injury")
    d = result_counts(data, "missed_match")
    bw = win_prob(data, "before_injury")
    dw = win_prob(data, "missed_match")
    change = f"{(dw - bw) * 100:+.1f} pp" if pd.notna(bw) and pd.notna(dw) else "n/a"
    metric_row([("Win rate before", pct(bw), "3 matches before the injury"),
                ("Win rate absent", pct(dw), "3 matches missed"),
                ("Change", change, "Percentage points"),
                ("Losses while absent", str(d["L"]), "Total in the data")])
    if pd.notna(bw) and pd.notna(dw):
        st.write(f"The win rate moves by {(dw - bw) * 100:+.1f} percentage points while the player is out. "
                 "That's an association in this data, not proof the absence caused it.")
    rec = pd.DataFrame({"Period": ["Before injury", "While absent"], "Wins": [b["W"], d["W"]],
                        "Draws": [b["D"], d["D"]], "Losses": [b["L"], d["L"]]})
    left, right = st.columns(2)
    with left:
        show(px.bar(rec, x="Period", y=["Wins", "Draws", "Losses"], barmode="group", text_auto=True,
                    title="Wins, draws and losses"))
    with right:
        prob = pd.DataFrame({"Period": ["Before injury", "While absent"], "Win rate": [bw, dw]}).dropna()
        fig = px.bar(prob, x="Period", y="Win rate", range_y=[0, 1], text="Win rate", title="Win rate")
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        show(fig)
    gd = pd.DataFrame({"Period": ["Before injury", "While absent", "After return"],
                       "Average GD": [data["Avg GD Before"].mean(), data["Avg GD During"].mean(), data["Avg GD After"].mean()]})
    show(px.line(gd, x="Period", y="Average GD", markers=True, title="Average goal difference over the injury period"), 360)


    insight_box("Takeaway", f"Win rate was <b>{pct(bw)}</b> before injury and <b>{pct(dw)}</b> while absent. This comparison describes the matches represented in the dataset; opponent strength, other absences and match context are not controlled for.")


@page
def rq3():
    header("RQ3: Form after coming back",
           "How does an individual player's performance change following their return from injury?")
    r = data[["Name", "Team Name", "Age", "Position", "Avg Rating Before", "Avg Rating After", "Rating Change"]] \
        .dropna(subset=["Avg Rating Before", "Avg Rating After"])
    if r.empty:
        st.info("Not enough valid before/after ratings.")
        return
    s = (r.groupby(["Name", "Team Name"], as_index=False)
         .agg(Before=("Avg Rating Before", "mean"), After=("Avg Rating After", "mean"), Change=("Rating Change", "mean"),
              Age=("Age", "first"), Position=("Position", "first"))
         .sort_values("Change", ascending=False))
    metric_row([("Players", f"{len(s):,}", "With ratings before and after"),
                ("Improved", f"{(s['Change'] > 0).mean():.0%}", "Share with a higher rating after"),
                ("Biggest improvement", str(s.iloc[0]["Name"]), f"+{s.iloc[0]['Change']:.2f} rating points"),
                ("Mean change", f"{s['Change'].mean():+.2f}", "Rating points")])
    left, right = st.columns(2)
    with left:
        fig = px.bar(s.head(12).sort_values("Change"), x="Change", y="Name", orientation="h", text="Change",
                     title="Biggest rating improvements")
        fig.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
        show(fig)
    with right:
        st.subheader("Look up a player")
        chosen = st.selectbox("Player", sorted(s["Name"].unique()), label_visibility="collapsed")
        p = s[s["Name"] == chosen].iloc[0]
        metric_row([("Before", f"{p['Before']:.2f}", "Avg rating"),
                    ("After", f"{p['After']:.2f}", "Avg rating"),
                    ("Change", f"{p['Change']:+.2f}", "After minus before")])
        if p["Change"] == 0:
            st.write(f"**{chosen}** stayed the same.")
        else:
            verb = "improved" if p["Change"] > 0 else "dropped"
            st.write(f"**{chosen}** {verb} by {abs(p['Change']):.2f} rating points.")
    show(px.histogram(s, x="Change", nbins=20, title="Spread of rating changes"), 360)


    improved = (s["Change"] > 0).mean() * 100
    insight_box("Recovery takeaway", f"<b>{improved:.1f}%</b> of players with usable ratings had a higher average rating after returning. The leaderboard shows rating change, not proof that injury or recovery alone caused the change.")


@page
def rq4():
    header("RQ4: Do injuries cluster?",
           "Are injuries concentrated during particular months or within specific clubs?")
    heat = data.dropna(subset=["Team Name", "Month Number"]).groupby(["Team Name", "Month Number"]).size().reset_index(name="Injuries")
    if heat.empty:
        st.info("Not enough date/club data.")
        return
    monthly = data["Injury Month"].value_counts()
    metric_row([("Busiest month", str(monthly.index[0]) if len(monthly) else "n/a",
                 f"{int(monthly.iloc[0]) if len(monthly) else 0} injuries"),
                ("Club with most injuries", str(data["Team Name"].value_counts().index[0]), "By record count"),
                ("Months covered", str(data["Month Number"].nunique()), "Months with at least one injury")])
    heat["Month"] = heat["Month Number"].astype(int).map(lambda m: MONTHS[m - 1])
    pvt = heat.pivot(index="Team Name", columns="Month", values="Injuries").fillna(0)
    pvt = pvt.reindex(columns=[m for m in MONTHS if m in pvt.columns])
    show(px.imshow(pvt, aspect="auto", text_auto=True, title="Injuries by club and month"), max(380, 22 * len(pvt) + 120))
    left, right = st.columns(2)
    with left:
        md = data.groupby("Month Number").size().reindex(range(1, 13), fill_value=0)
        z = pd.DataFrame({"Month": MONTHS, "Injuries": md.values})
        show(px.line(z, x="Month", y="Injuries", markers=True, title="Injuries by month"))
    with right:
        club_counts = data["Team Name"].value_counts().head(12).sort_values()
        z = club_counts.rename_axis("Club").reset_index(name="Injuries")
        show(px.bar(z, x="Injuries", y="Club", orientation="h", title="Clubs with the most injuries (top 12)"))


    peak_month = monthly.index[0] if len(monthly) else "n/a"
    insight_box("Cluster takeaway", f"<b>{peak_month}</b> is the most frequently recorded injury month in the current view. Use the heatmap to check whether the pattern is spread across clubs or driven by only one or two teams; missing injury dates can hide some cases.")


@page
def rq5():
    header("RQ5: Which clubs are hit hardest?",
           "Which clubs experience the greatest impact on team performance as a result of injuries?")
    club = (data.groupby("Team Name")
            .agg(Injuries=("Name", "size"), Avg_Duration=("Injury Duration", "mean"),
                 Avg_Drop=("Team Performance Drop", "mean"), Avg_FIFA=("FIFA rating", "mean"))
            .reset_index())
    club["Absence Win %"] = [win_prob(data[data["Team Name"] == x], "missed_match") for x in club["Team Name"]]
    club = club.sort_values("Avg_Drop", ascending=False)
    if club["Avg_Drop"].notna().sum() == 0:
        st.info("Not enough valid data for this view.")
        return
    metric_row([("Biggest drop", str(club.iloc[0]["Team Name"]), f"{club.iloc[0]['Avg_Drop']:.2f} goal difference"),
                ("Most injuries", str(club.sort_values("Injuries", ascending=False).iloc[0]["Team Name"]), f"{int(club['Injuries'].max())} records"),
                ("Clubs", str(len(club)), "In the current view")])
    left, right = st.columns(2)
    with left:
        fig = px.bar(club.dropna(subset=["Avg_Drop"]).head(12).sort_values("Avg_Drop"), x="Avg_Drop", y="Team Name",
                     orientation="h", text="Avg_Drop", title="Average drop by club (top 12)")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        show(fig)
    with right:
        z = club.dropna(subset=["Injuries", "Avg_Drop", "Avg_FIFA"])
        if len(z):
            show(px.scatter(z, x="Injuries", y="Avg_Drop", size="Avg_FIFA", hover_name="Team Name",
                            title="Injury count vs. average drop (bubble = avg FIFA rating)",
                            labels={"Injuries": "Injury records", "Avg_Drop": "Average drop"}))
    z = club.dropna(subset=["Avg_Duration", "Avg_Drop"]).copy()
    z["Avg_Duration"] = z["Avg_Duration"].clip(lower=0)
    if len(z):
        show(px.scatter(z, x="Avg_Duration", y="Avg_Drop", size="Injuries", hover_name="Team Name",
                        title="Average time out vs. average drop",
                        labels={"Avg_Duration": "Average duration (days)", "Avg_Drop": "Average drop"}), 380)
    st.subheader("All clubs")
    st.dataframe(club.style.format({"Avg_Duration": "{:.1f}", "Avg_Drop": "{:.2f}", "Avg_FIFA": "{:.2f}", "Absence Win %": "{:.1%}"}),
                 use_container_width=True, hide_index=True)


    worst = club.dropna(subset=["Avg_Drop"]).iloc[0]
    insight_box("Club-level takeaway", f"<b>{worst['Team Name']}</b> has the highest average performance drop in this filtered view ({worst['Avg_Drop']:.2f}). Compare this with injury volume and time out before concluding that a club is more vulnerable.")


@page
def rq6():
    header("RQ6: Does a player's FIFA rating matter?",
           "To what extent does a player's FIFA rating relate to the change in team performance during absence?")
    z = clean_plot(data, ["FIFA rating", "Team Performance Drop"])
    if len(z) < 3:
        st.info("Not enough valid numeric records.")
        return
    r = corr(z, "FIFA rating", "Team Performance Drop")
    strength = "Strong" if abs(r) >= .7 else "Moderate" if abs(r) >= .4 else "Weak"
    metric_row([("Correlation (r)", f"{r:.2f}", "Linear relationship, -1 to 1"),
                ("Records", f"{len(z):,}", "Rows with both values"),
                ("Strength", strength, "Based on |r|")])
    st.write("Correlation only tells us the two move together in this dataset. It doesn't show that a higher rating causes a bigger drop.")
    show(px.scatter(z, x="FIFA rating", y="Team Performance Drop", trendline=TREND, opacity=.65,
                    title=f"FIFA rating vs. team drop (r = {r:.2f})"))
    z["Rating band"] = pd.cut(z["FIFA rating"], bins=4)
    bins = z.groupby("Rating band", observed=True)["Team Performance Drop"].mean().reset_index()
    bins["Rating band"] = bins["Rating band"].astype(str)
    fig = px.bar(bins, x="Rating band", y="Team Performance Drop", text="Team Performance Drop",
                 title="Average drop by FIFA rating band")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    show(fig, 360)


    direction = "positive" if r > 0 else "negative" if r < 0 else "near zero"
    insight_box("Correlation takeaway", f"The relationship is <b>{strength.lower()} and {direction}</b> (r = {r:.2f}). FIFA rating alone explains only part of the variation, and correlation does not establish causation.")


@page
def rq7():
    header("RQ7: Win probability",
           "How does the probability of a team winning a match change when a player is absent?")
    bw = win_prob(data, "before_injury")
    dw = win_prob(data, "missed_match")
    aw = win_prob(data, "after_injury")
    change = f"{(dw - bw) * 100:+.1f} pp" if pd.notna(bw) and pd.notna(dw) else "n/a"
    metric_row([("Before", pct(bw), "Win rate"), ("During", pct(dw), "Win rate"),
                ("After", pct(aw), "Win rate"), ("Change", change, "Before to during")])
    if pd.notna(bw) and pd.notna(dw):
        st.write(f"Win rate changes by {(dw - bw) * 100:+.1f} percentage points while the player is absent, compared with before the injury.")
    z = pd.DataFrame({"Period": ["Before injury", "While absent", "After return"], "Win rate": [bw, dw, aw]}).dropna()
    fig = px.line(z, x="Period", y="Win rate", markers=True, text="Win rate", range_y=[0, 1],
                  title="Win rate across the injury period")
    fig.update_traces(texttemplate="%{text:.1%}", textposition="top center")
    show(fig, 360)
    b = result_counts(data, "before_injury")
    d = result_counts(data, "missed_match")
    a = result_counts(data, "after_injury")
    rec = pd.DataFrame({"Period": ["Before", "During", "After"], "Wins": [b["W"], d["W"], a["W"]],
                        "Draws": [b["D"], d["D"], a["D"]], "Losses": [b["L"], d["L"], a["L"]]})
    show(px.bar(rec, x="Period", y=["Wins", "Draws", "Losses"], barmode="group", text_auto=True,
                title="Match results by period"), 360)


    insight_box("Probability takeaway", f"Observed win rates: before <b>{pct(bw)}</b>, during absence <b>{pct(dw)}</b>, after return <b>{pct(aw)}</b>. These are descriptive proportions from the available match records, not a guarantee of the next match's outcome.")


@page
def rq8():
    header("RQ8: Age and recovery",
           "To what extent is player age associated with the change in performance following recovery?")
    z = clean_plot(data, ["Age", "Rating Change"])
    if len(z) < 3:
        st.info("Not enough valid age/recovery pairs.")
        return
    r = corr(z, "Age", "Rating Change")
    strength = "Strong" if abs(r) >= .7 else "Moderate" if abs(r) >= .4 else "Weak"
    metric_row([("Correlation (r)", f"{r:.2f}", "Age vs. rating change"),
                ("Records", f"{len(z):,}", "Rows with both values"),
                ("Strength", strength, "Based on |r|")])
    show(px.scatter(z, x="Age", y="Rating Change", trendline=TREND, opacity=.65,
                    title=f"Age vs. rating change after injury (r = {r:.2f})"))
    z["Age band"] = pd.cut(z["Age"], bins=[0, 20, 23, 26, 30, 100], labels=["20 and under", "21-23", "24-26", "27-30", "31+"])
    ageband = z.groupby("Age band", observed=True)["Rating Change"].mean().reset_index()
    fig = px.bar(ageband, x="Age band", y="Rating Change", text="Rating Change", title="Average rating change by age group")
    fig.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
    show(fig, 360)


MODEL_COLS = ["Age", "FIFA rating", "Position", "Injury", "Team Name", "Season", "Avg GD Before", "Avg Rating Before"]
NUM_COLS = ["Age", "FIFA rating", "Avg GD Before", "Avg Rating Before"]
CAT_COLS = ["Position", "Injury", "Team Name", "Season"]


def fit_model(source):
    
    required = MODEL_COLS + ["Team Performance Drop"]
    if any(c not in source.columns for c in required):
        return None
    md = source[required].copy().dropna(subset=["Team Performance Drop"])

    if len(md) < 30 or md["Team Performance Drop"].nunique() < 2:
        return None
    X, y = md[MODEL_COLS], md["Team Performance Drop"]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS)])
    pipe = Pipeline([("pre", pre), ("model", RandomForestRegressor(n_estimators=220, max_depth=8, min_samples_leaf=3, random_state=42))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    names = pipe.named_steps["pre"].get_feature_names_out()
    imps = pipe.named_steps["model"].feature_importances_
    return {"pipe": pipe, "mae": mean_absolute_error(yte, pred), "r2": r2_score(yte, pred),
            "n_train": len(Xtr), "names": names, "imps": imps}


@st.cache_resource(show_spinner="Training the model...")
def train_model(_source):
    
    return fit_model(_source)


def med(col, default=0.0):
    v = df[col].median()
    return float(v) if pd.notna(v) else default


    direction = "positive" if r > 0 else "negative" if r < 0 else "near zero"
    insight_box("Age and recovery takeaway", f"Age and rating change show a <b>{strength.lower()} {direction} association</b> (r = {r:.2f}) in this view. The scatter's spread matters too: age is only one of many factors affecting performance after return.")


@page
def tools():
    header("Tools and prediction", "Extra tools beyond the eight research questions. These use the full dataset, not the sidebar filters.")

    st.subheader("Player lookup")
    players = sorted(df["Name"].dropna().astype(str).unique())
    chosen = st.selectbox("Player", players, key="tool_player")
    p = df[df["Name"] == chosen]
    metric_row([
        ("Injury records", str(len(p)), "In the dataset"),
        ("Avg time out", f"{p['Injury Duration'].mean():.0f} days" if p["Injury Duration"].notna().any() else "n/a", "Days"),
        ("Avg team drop", f"{p['Team Performance Drop'].mean():.2f}" if p["Team Performance Drop"].notna().any() else "n/a", "Goal difference"),
        ("Rating change", f"{p['Rating Change'].mean():+.2f}" if p["Rating Change"].notna().any() else "n/a", "After minus before"),
    ])
    st.dataframe(p[["Name", "Team Name", "Position", "Age", "Season", "Injury", "Date of Injury", "Date of return",
                    "Injury Duration", "Team Performance Drop", "Rating Change"]],
                 use_container_width=True, hide_index=True)

    st.subheader("Compare two clubs")
    clubs = sorted(df["Team Name"].dropna().astype(str).unique())
    a, b = st.columns(2)
    with a:
        ca = st.selectbox("Club A", clubs, key="tool_ca")
    with b:
        cb = st.selectbox("Club B", clubs, index=min(1, len(clubs) - 1), key="tool_cb")
    comp = []
    for c in [ca, cb]:
        q = df[df["Team Name"] == c]
        comp.append({"Club": c, "Injuries": len(q), "Avg duration": q["Injury Duration"].mean(),
                     "Avg drop": q["Team Performance Drop"].mean(), "Absence win %": win_prob(q, "missed_match")})
    st.dataframe(pd.DataFrame(comp).style.format({"Avg duration": "{:.1f}", "Avg drop": "{:.2f}", "Absence win %": "{:.1%}"}),
                 use_container_width=True, hide_index=True)

    st.subheader("Predict the team impact of an injury")
    st.info("Based on past patterns in this dataset. It isn't a medical forecast or a guarantee about future matches.")
    m = train_model(df)
    if m is None:
        st.info("Not enough valid records to train the prediction model.")
        return
    metric_row([("Test MAE", f"{m['mae']:.2f}", "Mean absolute error in goal-difference units; lower is better"),
                ("Test R²", f"{m['r2']:.2f}", "Held-out test score; can be negative if predictions are poor"),
                ("Training rows", f"{m['n_train']:,}", "80% of eligible records")])
    if m["r2"] < 0:
        st.warning("The current test score is below a simple mean baseline. Treat individual predictions as exploratory, not reliable forecasts.")
    elif m["r2"] < 0.3:
        st.warning("The model has limited predictive power on the held-out data. Use the result cautiously.")
    else:
        st.caption("The model shows some predictive signal on the held-out split, but results are estimates from historical records—not certainty about future matches.")
    left, right = st.columns(2, gap="large")
    with left:
        pa = st.number_input("Age", float(df["Age"].min()), float(df["Age"].max()), med("Age", 25.0), 1.0, key="tpa")
        pf = st.number_input("FIFA rating", float(df["FIFA rating"].min()), float(df["FIFA rating"].max()),
                             med("FIFA rating", 75.0), .1, key="tpf")
        pp = st.selectbox("Position", sorted(df["Position"].dropna().astype(str).unique()), key="tpp")
        pi = st.selectbox("Injury", sorted(df["Injury"].dropna().astype(str).unique()), key="tpi")
        pc = st.selectbox("Club", clubs, key="tpc")
        ps = st.selectbox("Season", sorted(df["Season"].dropna().astype(str).unique()), key="tps")
        pg = st.number_input("Avg goal difference before", value=med("Avg GD Before"), step=.1, key="tpg")
        pr = st.number_input("Avg rating before", value=med("Avg Rating Before", 6.5), step=.1, key="tpr")
        row = pd.DataFrame([{"Age": pa, "FIFA rating": pf, "Position": pp, "Injury": pi, "Team Name": pc,
                             "Season": ps, "Avg GD Before": pg, "Avg Rating Before": pr}])
        est = float(m["pipe"].predict(row)[0])
        hist = df["Team Performance Drop"].dropna()
        pctile = float((hist <= est).mean())
        metric_row([("Predicted drop", f"{est:.2f}", "Goal difference"),
                    ("Percentile", f"{pctile:.0%}", "Where this sits in the historical data"),
                    ("Impact level", "High" if est > .75 else "Moderate" if est > .25 else "Low", "Same cutoffs as the Impact level filter")])
    with right:
        fi = pd.DataFrame({"Feature": m["names"], "Importance": m["imps"]}).nlargest(12, "Importance")
        fi["Feature"] = fi["Feature"].str.replace(r"^(num__|cat__)", "", regex=True)
        show(px.bar(fi.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                    title="What the model relies on most"), 520)


    insight_box("Using the prediction responsibly", "Use the model to explore patterns in this dataset, not as a certain forecast. The test MAE describes typical prediction error in goal-difference units; a low or negative R² can mean the available features do not predict the outcome reliably.")


@page
def data_page():
    header("Data and methodology", "How the numbers are built, and a look at the underlying data.")
    metric_row([("Raw records", f"{len(df):,}", "In the CSV"), ("Filtered", f"{len(data):,}", "Current view"),
                ("Players", f"{df['Name'].nunique():,}", "Unique"), ("Clubs", f"{df['Team Name'].nunique():,}", "Unique")])
    st.subheader("Missing values")
    cols = ["Name", "Team Name", "Position", "Age", "FIFA rating", "Injury", "Date of Injury", "Date of return",
            "Team Performance Drop", "Avg Rating Before", "Avg Rating After"]
    dq = pd.DataFrame({"Field": [c for c in cols if c in data.columns],
                       "Missing": [int(data[c].isna().sum()) for c in cols if c in data.columns]})
    dq["Missing %"] = dq["Missing"] / max(len(data), 1)
    st.dataframe(dq.style.format({"Missing %": "{:.1%}"}), use_container_width=True, hide_index=True)
    st.subheader("How things are calculated")
    st.markdown("""
- **Injury duration** = return date - injury date, in days. Rows where the return date is before the injury date are treated as missing.
- **Average GD before / during / after** = mean goal difference over the three matches in each window.
- **Team performance drop** = average GD before - average GD during.
- **Rating change** = average player rating after - average player rating before.
- **Win rate** = wins / matches observed.
- **Correlation** shows whether two things move together. It doesn't show cause and effect.
- **Impact level** (high / moderate / low) is just a bucket of the performance drop. It says nothing about how serious the injury was medically.
""")
    st.subheader("Data preview")
    show_cols = [c for c in cols if c in data.columns]
    st.dataframe(data[show_cols].head(500), use_container_width=True, hide_index=True)
    st.caption("First 500 rows only.")
    st.download_button("Download filtered data (CSV)", data.to_csv(index=False).encode("utf-8"),
                       "footlens_filtered_data.csv", "text/csv")


    insight_box("Data quality note", "Results depend on which fields are recorded. Missing dates or ratings reduce the usable sample for some charts, and the before/during/after windows cover only the three matches supplied for each period.")


df = init()
data = sidebar_filters(df)

pages = [
    st.Page(home, title="Home", default=True),
    st.Page(rq1, title="Injury impact"),
    st.Page(rq2, title="Team absence"),
    st.Page(rq3, title="Player recovery"),
    st.Page(rq4, title="Injury clusters"),
    st.Page(rq5, title="Club impact"),
    st.Page(rq6, title="FIFA rating"),
    st.Page(rq7, title="Win probability"),
    st.Page(rq8, title="Age and recovery"),
    st.Page(tools, title="Tools"),
    st.Page(data_page, title="Data"),
]

pg = st.navigation(pages, position="top")
pg.run()

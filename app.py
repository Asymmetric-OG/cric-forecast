import streamlit as st
import pandas as pd
import json
import base64
import plotly.express as px

from src.inference import load_models, generate_match_predictions, calculate_player_risk

#THEME
st.set_page_config(page_title="Fantasy Score Prediction Dashboard", layout="wide", initial_sidebar_state="expanded")

def apply_theme(main_bg, sidebar_bg, banner_img):
    try:
        with open(main_bg, "rb") as f: main_encoded = base64.b64encode(f.read()).decode()
        st.markdown(f"""<style>.stApp {{ background: linear-gradient(rgba(10,10,20,0.55), rgba(10,10,20,0.70)), url("data:image/png;base64,{main_encoded}"); background-size: cover; background-position: center; background-attachment: fixed; filter: brightness(1.15) contrast(1.05); }}</style>""", unsafe_allow_html=True)
    except FileNotFoundError: pass

    try:
        with open(sidebar_bg, "rb") as f: side_encoded = base64.b64encode(f.read()).decode()
        st.markdown(f"""<style>section[data-testid="stSidebar"] {{ background: linear-gradient(rgba(18,18,30,0.65), rgba(18,18,30,0.85)), url("data:image/png;base64,{side_encoded}"); background-size: cover; background-position: center; filter: brightness(1.18) contrast(1.05); border-right: 1px solid rgba(255,255,255,0.08); }} section[data-testid="stSidebar"] > div:first-child {{ backdrop-filter: blur(8px); padding-top: 10px; }} section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{ color: #ffffff; font-weight: 600; }} section[data-testid="stSidebar"] label {{ color: #dcdcf0 !important; font-weight: 500; font-size: 14px; }}</style>""", unsafe_allow_html=True)
    except FileNotFoundError: pass

    try:
        with open(banner_img, "rb") as f: banner_encoded = base64.b64encode(f.read()).decode()
        st.markdown(f"""<style>.header-banner {{ background: linear-gradient(rgba(0,0,0,0.25), rgba(0,0,0,0.40)), url("data:image/png;base64,{banner_encoded}"); background-size: cover; background-position: center; padding: 40px; border-radius: 12px; text-align: center; color: white; margin-bottom: 20px; }}</style><div class="header-banner"><h1>🏏 IPL Fantasy Score Prediction</h1><p>AI-Powered insights on player form, risk, and consistency</p></div>""", unsafe_allow_html=True)
    except FileNotFoundError:
        st.title("🏏 IPL Fantasy Score Prediction")

apply_theme("backgound.png", "bkg.png", "header.png")
st.markdown("""<style>h1, h2, h3, h4, h5, h6, p, label, span { color: #f0f0f5; text-shadow: 0px 1px 3px rgba(0,0,0,0.6); } .stDataFrame, .stTable { background-color: rgba(20,20,30,0.85); border-radius: 8px; padding: 5px; } .block-container { padding-top: 0rem; } header[data-testid="stHeader"] { display: none; }</style>""", unsafe_allow_html=True)

# Load models, static data

@st.cache_data
def get_static_data():
    hist_df = pd.read_csv("data/t20_master_aggregated.csv")
    hist_df["player"] = hist_df["player"].astype(str).str.strip()
    with open("data/team_registry.json", "r") as f: team_reg = json.load(f)
    with open("data/player_registry.json", "r") as f: player_reg = json.load(f)
    with open("data/scalers.json", "r") as f: scalers = json.load(f)
    return hist_df, team_reg, player_reg, scalers

@st.cache_resource
def get_ml_models():
    return load_models()

loading_placeholder = st.empty()

with loading_placeholder.container():
    st.markdown("""
        <div style='text-align: center; padding: 50px; background-color: rgba(20,20,30,0.8); border-radius: 10px; margin-top: 50px;'>
            <h2 style='color: #4CAF50;'>⚙️ Waking up the AI...</h2>
            <p>Loading, please wait while the PyTorch models initialize. This takes about 30-60 seconds on the very first run, but will be lightning fast afterward!</p>
        </div>
    """, unsafe_allow_html=True)
    
with st.spinner("Loading models into RAM..."):
    hist_df, team_registry, player_registry, scalers = get_static_data()
    models = get_ml_models()

loading_placeholder.empty()
teams = list(team_registry.keys())

# User Interface
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏏 Match Dashboard","🔍 Player Insights"])
st.sidebar.header("Match Selection")
team1 = st.sidebar.selectbox("Team 1", teams, index=0)
team2 = st.sidebar.selectbox("Team 2", teams, index=1)

if "live_predictions" not in st.session_state: st.session_state.live_predictions = None

if st.sidebar.button("Predict Match via AI"):
    if team1 == team2: st.sidebar.error("Select two different teams!")
    else:
        with st.spinner("Running PyTorch Models..."):
            raw_preds = generate_match_predictions(team1, team2, hist_df, team_registry, player_registry, scalers, models)
            raw_preds.rename(columns={"fp": "pred_total_fp"}, inplace=True)
            st.session_state.live_predictions = calculate_player_risk(raw_preds, hist_df)
            st.sidebar.success("Predictions generated!")

match_df = st.session_state.live_predictions

# Match dashboard section
if page == "🏏 Match Dashboard":
    if match_df is None: st.info("👈 Select two teams and click 'Predict Match via AI' to begin.")
    else:
        match_df = match_df.sort_values("pred_total_fp", ascending=False)
        st.subheader("🏆 Best Predicted XI")
        st.dataframe(match_df.head(11)[["player_name", "pred_total_fp", "consistency", "risk_tag"]], use_container_width=True)

        st.subheader("⚠️ Risk Profile")
        risk_counts = match_df["risk_tag"].value_counts().reset_index()
        fig = px.bar(risk_counts, x="risk_tag", y="count", color="risk_tag", color_discrete_map={"Safe": "#4CAF50", "Moderate": "#FFC107", "High Risk": "#FF5252"}, text="count")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Players", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("### ✅ Safe Picks")
            st.dataframe(match_df[(match_df["risk_tag"] == "Safe") & (match_df["pred_total_fp"] > 40)][["player_name", "pred_total_fp"]])
        with col2:
            st.write("### 🎯 High-Risk High-Reward")
            st.dataframe(match_df[(match_df["risk_tag"] == "High Risk") & (match_df["pred_total_fp"] > 60)][["player_name", "pred_total_fp"]])

# Player dashboard section
elif page == "🔍 Player Insights":
    if match_df is None: st.info("👈 Select two teams and click 'Predict Match via AI' to begin.")
    else:
        name_to_id = dict(zip(match_df["player_name"], match_df["player_id"]))
        selected_name = st.selectbox("Select Player", sorted(match_df["player_name"].unique()))
        player_id = name_to_id[selected_name]
        p = match_df[match_df["player_id"] == player_id].iloc[0]

        st.markdown("### 🧑‍💻 Player Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Fantasy", f"{p['pred_total_fp']:.2f}")
        c2.metric("Consistency (CV)", f"{p['consistency']:.2f}")
        risk_color = "#4CAF50" if p["risk_tag"]=="Safe" else "#FFC107" if p["risk_tag"]=="Moderate" else "#FF5252"
        c3.markdown(f"<div style='padding:8px;border-radius:8px;background:{risk_color};text-align:center;font-weight:600;color:white;'>Risk: {p['risk_tag']}</div>", unsafe_allow_html=True)
        st.markdown("---")

        player_hist = hist_df[hist_df["player"] == selected_name].sort_values("match_id")
        if len(player_hist) > 5:
            last20 = player_hist.tail(20)
            avg_runs, avg_wkts = last20["runs"].mean(), last20["wickets"].mean()
            role = "Batsman" if avg_runs > 20 and avg_wkts < 0.5 else "Bowler" if avg_wkts > 1.0 and avg_runs < 15 else "All-Rounder" if avg_runs > 15 and avg_wkts > 0.5 else "Mixed"
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Matches Analyzed", len(player_hist))
            c2.metric("Avg Fantasy", f"{last20['fp'].mean():.1f}")
            c3.metric("Recent Form", f"{player_hist.tail(5)['fp'].mean():.1f}")
            c4.metric("Player Role", role)
            
            st.markdown("### 📈 Fantasy Performance Trend")
            chart_data = last20.copy()
            chart_data['Match'] = range(1, len(chart_data) + 1)
            st.line_chart(chart_data.set_index("Match")["fp"], use_container_width=True)
        else:
            st.info(f"Not enough historical data (Needs >5 matches. Currently has {len(player_hist)}).")
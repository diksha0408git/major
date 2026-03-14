#  IMPORTS 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import google.generativeai as genai
from statsmodels.tsa.arima.model import ARIMA

# PAGE CONFIG 
st.set_page_config(
    page_title="Healthcare Analytics with Forecast",
    page_icon="🏥",
    layout="wide"
)

# THEME COLORS 
PRIMARY_COLOR = "#10b981" # Emerald Green
SECONDARY_COLOR = "#0d2e33" # Deep Teal
BG_COLOR = "#f8fafc"
CHART_COLORS = ['#10b981', '#0d2e33', '#3b82f6', '#f59e0b', '#ef4444']

# SESSION STATE 
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "hospital" not in st.session_state:
    st.session_state["hospital"] = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# GEMINI API CONFIG 

try:
    genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", ""))
    
    # 1. Ask Google's servers for every model your key is allowed to use
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    
    # 2. Automatically pick the best available text model
    if available_models:
        # Prioritize 'flash' models for speed, otherwise just grab the first valid one
        chosen_model = next((m for m in available_models if "flash" in m), available_models[0])
        model = genai.GenerativeModel(chosen_model)
        api_configured = True
    else:
        api_configured = False
        
except Exception as e:
    api_configured = False
    st.error(f"Failed to connect to AI: {e}")

# GLOBAL CSS THEME
def apply_global_theme():
    st.markdown(f"""
        <style>
        .stApp {{ background-color: {BG_COLOR}; }}
        [data-testid="stSidebar"] {{ background-color: {SECONDARY_COLOR} !important; }}
        [data-testid="stSidebar"] * {{ color: white !important; }}
        div.stMetric {{ 
            background-color: white; 
            padding: 15px; 
            border-radius: 10px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.05); 
            border-left: 5px solid {PRIMARY_COLOR}; 
        }}
        .stButton>button {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border-radius: 8px;
            border: none;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: #059669;
            box-shadow: 0 5px 15px rgba(16, 185, 129, 0.3);
        }}
        </style>
    """, unsafe_allow_html=True)

# DYNAMIC LOGIN PAGE 
def login_page():
    st.markdown(f"""
        <style>
        /* Dark gradient background */
        .stApp {{ background: linear-gradient(135deg, #0f172a 0%, #0d2e33 100%); }}
        
        /* Bright white title */
        .main-title {{ 
            color: #ffffff !important; 
            font-size: 36px; 
            font-weight: 800; 
            text-align: center; 
            margin-bottom: 5px; 
            font-family: 'Segoe UI', sans-serif;
        }}
        
        /* Light grey subtitle */
        .sub-title {{ 
            color: #cbd5e1 !important; 
            font-size: 16px; 
            text-align: center; 
            margin-bottom: 25px; 
        }}
        
        /* White Card Form */
        [data-testid="stForm"] {{
            background-color: #ffffff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
            border: none;
        }}
        
        /* Login Button */
        [data-testid="stFormSubmitButton"] > button {{
            width: 100%;
            background-color: #10b981;
            color: white;
            border-radius: 8px;
            border: none;
            font-weight: bold;
            padding: 0.6rem;
            transition: all 0.3s ease;
        }}
        [data-testid="stFormSubmitButton"] > button:hover {{
            background-color: #059669;
            color: white;
            transform: translateY(-2px);
        }}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 1.2, 1])

    with col2:
        # NEW PROFESSIONAL HEALTHCARE IMAGE ADDED HERE
        
        st.image("dataset/imggifproject.gif", use_container_width=True)
        
        st.markdown("<h2 class='main-title'>Healthcare Analytics</h2>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'>Predictive insights for modern hospital management</p>", unsafe_allow_html=True)

        with st.form("login_form"):
            st.markdown("#### Login")
            username = st.text_input("Username", placeholder="e.g., admin")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            hospital = st.selectbox("Assigning Hospital", ["Hospital1", "Hospital2"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            login_btn = st.form_submit_button("Access Dashboard")

        if login_btn:
            if username == "admin" and password == "admin123":
                st.session_state["logged_in"] = True
                st.session_state["hospital"] = hospital
                st.rerun()
            else:
                st.error("Invalid Credentials. Please check and try again.")

# Show login if not logged in
if not st.session_state["logged_in"]:
    login_page()
    st.stop()

# Apply the theme to the rest of the app once logged in
apply_global_theme()

# LOAD DATA
@st.cache_data
def load_data(hospital_name):
    try:
        if hospital_name == "Hospital1":
            df = pd.read_csv("dataset/appointments_cleaned.csv")
            table_name = "appointments"
        else:
            df = pd.read_csv("dataset/patients_cleaned.csv")
            table_name = "patients"
        return df, table_name
    except FileNotFoundError:
        st.error(f"⚠️ Dataset not found! Please ensure your CSV files are securely placed inside the 'dataset/' folder.")
        st.stop()

df, table_name = load_data(st.session_state.hospital)

# SQLITE 
conn = sqlite3.connect("hospital_major_project.db", check_same_thread=False)
df.to_sql(table_name, conn, if_exists="replace", index=False)

def load_from_db(table):
    return pd.read_sql(f"SELECT * FROM {table}", conn)

# SIDEBAR 
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=50)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Visualizations", "Correlation", "Forecasting", "AI Chatbot", "Database"])

st.sidebar.markdown("---")
st.sidebar.write("Hospital Node:", f"**{st.session_state['hospital']}**")

if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    st.session_state["hospital"] = None
    st.rerun()

#  DASHBOARD 
if page == "Overview":
    st.title("Hospital Overview")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(df))
    c2.metric("Departments", df["department"].nunique() if "department" in df.columns else "N/A")
    c3.metric("Average Age", round(df["age"].mean(),1) if "age" in df.columns else "N/A")
    c4.metric("Beds Available", df["bed_availability"].sum() if "bed_availability" in df.columns else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Dataset Structure")
    clean_df = df.dropna()
    col1, col2 = st.columns(2)
    with col1: st.write("**Dataset Shape:**", df.shape)
    with col2: st.write("**After Cleaning:**", clean_df.shape)

    with st.expander("View Cleaned Data"):
        st.dataframe(clean_df.head(), use_container_width=True)

# VISUALIZATIONS 
elif page == "Visualizations":
    st.title("Analytics Visualizations")

    col1, col2 = st.columns(2)
    
    if "department" in df.columns:
        with col1:
            fig = px.pie(df, names="department", title="Department Distribution", color_discrete_sequence=CHART_COLORS)
            st.plotly_chart(fig, use_container_width=True)
            
    if "gender" in df.columns:
        with col2:
            fig = px.bar(df, x="gender", title="Gender Distribution", color="gender", color_discrete_sequence=CHART_COLORS)
            st.plotly_chart(fig, use_container_width=True)
    elif "status" in df.columns:
        with col2:
            fig = px.pie(df, names="status", title="Appointment Status", color_discrete_sequence=CHART_COLORS)
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    if "age" in df.columns:
        with col3:
            fig = px.histogram(df, x="age", nbins=30, title="Age Distribution", color_discrete_sequence=[PRIMARY_COLOR])
            st.plotly_chart(fig, use_container_width=True)

    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols:
        date_col = date_cols[0]
        daily = df.groupby(date_col).size().reset_index(name="count")
        with col4:
            fig = px.line(daily, x=date_col, y="count", title="Daily Trend", color_discrete_sequence=[SECONDARY_COLOR])
            st.plotly_chart(fig, use_container_width=True)

# CORRELATION 
elif page == "Correlation":
    st.title("Correlation Matrix")
    corr = df.select_dtypes(include=np.number).corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Numerical Feature Correlation", color_continuous_scale="Tealgrn")
    st.plotly_chart(fig, use_container_width=True)

# FORECASTING 
elif page == "Forecasting":
    st.title("Daily Admissions Forecast")
    date_cols = [c for c in df.columns if "date" in c.lower()]

    if not date_cols:
        st.warning("No date column found for forecasting.")
    else:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col])
        daily = df.groupby(date_col).size().reset_index(name="admissions")

        if len(daily) < 20:
            st.warning("Not enough data points for ARIMA forecasting.")
        else:
            model = ARIMA(daily["admissions"], order=(1,1,1))
            model_fit = model.fit()

            forecast_steps = st.slider("Select Forecast Horizon (Days)", 7, 30, 10)
            forecast = model_fit.forecast(steps=forecast_steps)
            future_dates = pd.date_range(start=daily[date_col].max(), periods=forecast_steps+1, freq="D")[1:]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily[date_col], y=daily["admissions"], mode='lines', name='Actual', line=dict(color=SECONDARY_COLOR)))
            fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines+markers', name='Forecast', line=dict(color=PRIMARY_COLOR, dash='dash')))
            fig.update_layout(title="Admissions Forecast (ARIMA)", xaxis_title="Date", yaxis_title="Admissions", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

# AI CHATBOT 

elif page == "AI Chatbot":
    st.title("🤖 Clara - Hospital AI Assistant")
    
    if not api_configured:
        st.warning("⚠️ Gemini API Key not found. Please add `GEMINI_API_KEY = 'your_key_here'` to `.streamlit/secrets.toml`.")
    else:
        st.info(f"Hi! I'm Clara. Ask me about data patterns, hospital operations, or bed occupancy metrics for {st.session_state.hospital}!")
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("E.g., What is the average age of our patients?"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            system_context = f"""
            You are Clara, a helpful and professional data analytics assistant for a Streamlit hospital dashboard.
            The user is looking at data for: {st.session_state.hospital}.
            
            Here is the summary of the current dataset they are analyzing:
            - Total Records: {len(df)}
            - Columns available: {', '.join(df.columns)}
            
            Statistical Summary of the data:
            {df.describe().to_markdown()}
            
            Please answer the following user query accurately based ONLY on the data summary provided above. 
            Keep your answer concise, professional, and do not show the raw markdown tables to the user.
            
            User Query: {prompt}
            """

            with st.chat_message("assistant"):
                try:
                    response = model.generate_content(system_context)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"API Error: {e}")
                    
# DATABASE 
elif page == "Database":
    st.title("SQLite Database View")
    st.dataframe(load_from_db(table_name), use_container_width=True)

# FOOTER 
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown(f"""
<style>
.footer {{
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: {SECONDARY_COLOR};
    color: white;
    text-align: center;
    padding: 10px;
    font-size: 13px;
    z-index: 999;
}}
</style>
<div class="footer">
    Healthcare Analytics Dashboard | Facility: <b>{st.session_state.hospital}</b> | © 2026 Insha Farhan & Diksha Tiwari
</div>
""", unsafe_allow_html=True)





































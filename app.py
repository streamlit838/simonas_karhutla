# Import library
import streamlit as st
import pandas as pd
import cgi
import matplotlib.pyplot as plt
from PIL import Image
import datetime
import plotly.express as px
from wordcloud import WordCloud
from dashboard import get_dashboard
from text_analyzer import get_text_analyzer
from monitoring import get_monitoring

# Set web page layouts
st.set_page_config(
    page_title="SIMONAS KARHUTLA", page_icon="assets/app_icon.png", layout="wide"
)

# Adding sidebar image
# st.sidebar.image(Image.open("assets/app_icon.png"), use_column_width=True)
st.sidebar.image(Image.open("assets/app_icon.png"), use_container_width=True)
# Adding sidebar title
st.sidebar.markdown(
    """
    <style>
        .centered-text {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            margin-top: -35px;
            font-size: 24px;
            margin-bottom: 10px; 
        }
        .title1-container {
            text-align: center;
        }
        .title2-container {
            text-align: center;
            margin-top: -35px;
        }
    </style>
    <div class='centered-text'>
        <div class='title1-container'>
            <h1>SIMONAS KARHUTLA</h1>
        </div>
        <div class='title2-container'>
            <h1>(Sistem Informasi Monitoring Analisis Sentimen Kebakaran Hutan dan Lahan)</h1>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

# Set activities selection options
act_opt = ["Dashboard", "Monitoring", "Text Analyzer"]
activities = st.sidebar.selectbox("Choose Your Activity", act_opt)

# Display web apps
if activities == "Dashboard":
    st.cache_data(get_dashboard())
elif activities == "Monitoring":
    st.cache_data(get_monitoring())
else:
    st.cache_data(get_text_analyzer())

# Let's remove the footer watermark
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            #GithubIcon {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Set the footer style
footer = """
<div style='text-align: center; font-size: 15px; padding: 10px;'>
    Copyright 2023 SIMONAS KARHUTLA - IPB University. All rights reserved
</div>
"""
# Display the footer
st.markdown(footer, unsafe_allow_html=True)

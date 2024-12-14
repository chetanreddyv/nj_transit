import streamlit as st
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="NJ Transit Smart Assistant",
    page_icon="🚆",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .banner-container {
        width: 100%;
        margin-bottom: 1rem;
    }
    .banner-image {
        width: 100%;
        object-fit: cover;
        max-height: 200px;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E90FF;
        margin: 1rem 0 2rem 0;
        padding: 0.5rem;
        border-bottom: 3px solid #1E90FF;
    }
    .feature-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #e9ecef;
    }
    .highlight-text {
        color: #1E90FF;
        font-weight: bold;
    }
    .description-text {
        font-size: 1.1rem;
        line-height: 1.6;
        margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and display the logo as a banner
logo_path = "assets/new_jesry_transit_logo.png"
logo = Image.open(logo_path)

# Banner and Title
st.markdown('<div class="banner-container">', unsafe_allow_html=True)
st.image(logo, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Smart Journey Assistant 🚆</h1>', unsafe_allow_html=True)

# Introduction
st.markdown("""
    <div class="description-text">
    Welcome to the future of transit planning! Our smart assistant combines advanced machine learning 
    with intuitive design to make your NJ Transit journey smoother than ever.
    </div>
    """, unsafe_allow_html=True)

# Main Features Section
st.markdown("## 🌟 What We Offer")

# Delay Prediction Feature
st.markdown("""
    <div class="feature-card">
    <h3>🕒 Smart Delay Prediction</h3>
    <p class="description-text">
    Our machine learning model analyzes historical transit data to provide accurate delay predictions:
    <ul>
        <li>Trained on extensive NJ Transit historical data</li>
        <li>Considers factors like time of day, day of week, and route specifics</li>
        <li>Provides real-time delay estimates for better journey planning</li>
        <li>Accuracy rate of over 85% in predicting delays</li>
    </ul>
    </p>
    </div>
    """, unsafe_allow_html=True)
# Mechanical Failure Analysis Feature
st.markdown("""
    <div class="feature-card">
    <h3>🔧 Mechanical Failure Analysis</h3>
    <p class="description-text">
    Advanced diagnostics and predictive maintenance insights:
    <ul>
        <li>Monitoring of train mechanical systems by month and year</li>
        <li>Predictive analytics for potential mechanical failures</li>
        <li>Historical maintenance data analysis</li>
        <li>Proactive maintenance recommendations to prevent delays</li>
    </ul>
    </p>
    </div>
    """, unsafe_allow_html=True)
# AI Support Feature
st.markdown("""
    <div class="feature-card">
    <h3>💬 AI Customer Support</h3>
    <p class="description-text">
    Get instant answers with our AI-powered customer support:
    <ul>
        <li>24/7 availability for your questions</li>
        <li>Trained on NJ Transit-specific information</li>
        <li>Handles queries about schedules, fares, routes, and more</li>
        <li>Natural conversation flow for better user experience</li>
    </ul>
    </p>
    </div>
    """, unsafe_allow_html=True)

# Project Background
st.markdown("## 🎯 Project Background")
st.markdown("""
    <div class="description-text">
    This project was developed during the Rutgers University Hackathon, aiming to improve the commuter 
    experience on NJ Transit. By combining data science with user-centric design, we've created a 
    tool that helps passengers make informed decisions about their travel plans.
    </div>
    """, unsafe_allow_html=True)

# How to Use
st.markdown("## 📝 How to Navigate")
st.markdown("""
    <div class="description-text">
    Use the sidebar to access our main features:
    <ul>
        <li><span class="highlight-text">Delay Prediction:</span> Get accurate delay forecasts for your journey</li>
        <li><span class="highlight-text">AI Support:</span> Chat with our AI assistant for instant help</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
    Developed with ❤️ at Rutgers University Hackathon
    </div>
    """, unsafe_allow_html=True)
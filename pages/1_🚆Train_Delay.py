import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="NJ Transit Rail Delay Prediction",
    page_icon="🚆",
    layout="wide"
)

# Custom CSS for responsive title
st.markdown("""
    <style>
    .responsive-title {
        font-size: calc(1.5rem + 1.5vw);
        font-weight: bold;
        line-height: 1.2;
        padding-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load logo and create header layout
logo_path = "assets/new_jesry_transit_logo.png"
logo = Image.open(logo_path)

# Create two columns for logo and title
col1, col2 = st.columns([1, 3])

with col1:
    # Display logo with new parameter
    st.image(logo, use_container_width=True)

with col2:
    st.markdown('<p class="responsive-title">NJ Transit Rail Delay Prediction 🚆</p>', unsafe_allow_html=True)

# Define model performance metrics
metrics = {
    'Mean Absolute Error (MAE)': 1.87,
    'Root Mean Squared Error (RMSE)': 4.67
}

# Add sidebar section with metrics and visual indicators
st.sidebar.markdown("### Model Performance Metrics")

# Display metrics with custom styling
st.sidebar.markdown("""
<style>
    .metric-container {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        background-color: #f0f2f6;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 14px;
        color: #424242;
    }
</style>
""", unsafe_allow_html=True)

# Display each metric with custom formatting
for metric_name, value in metrics.items():
    st.sidebar.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">{metric_name}</div>
        <div class="metric-value">{value:.2f} minutes</div>
    </div>
    """, unsafe_allow_html=True)

# Add interpretation
st.sidebar.markdown("""
---
**Model Accuracy Interpretation:**
- MAE of 1.87 minutes indicates average prediction error
- RMSE of 4.67 minutes shows spread of prediction errors
""")

# Station dictionary
stations = {'Newark Penn Station': 107, 'Union': 38105, 'Roselle Park': 31, 'Cranford': 32, 'Westfield': 155, 'Fanwood': 44, 'Netherwood': 102, 'Plainfield': 120, 'Dunellen': 36, 'Bound Brook': 21, 'Bridgewater': 24, 'Somerville': 138, 'New York Penn Station': 105, 'Secaucus Upper Lvl': 38187, 'Newark Airport': 37953, 'Elizabeth': 41, 'Linden': 70, 'Rahway': 127, 'Metropark': 83, 'Metuchen': 84, 'Edison': 38, 'New Brunswick': 103, 'Princeton Junction': 125, 'Hamilton': 32905, 'Philadelphia': 1, 'Trenton': 148, 'Princeton': 124, 'North Elizabeth': 109, 'Avenel': 11, 'Woodbridge': 158, 'Perth Amboy': 119, 'South Amboy': 139, 'Aberdeen-Matawan': 37169, 'Hazlet': 59, 'Middletown NJ': 85, 'Red Bank': 130, 'Little Silver': 73, 'Hoboken': 63, 'Secaucus Lower Lvl': 38174, 'Wood Ridge': 160, 'Teterboro': 146, 'Essex Street': 43, 'Anderson Street': 5, 'New Bridge Landing': 110, 'River Edge': 132, 'Oradell': 111, 'Emerson': 42, 'Westwood': 156, 'Hillsdale': 62, 'Woodcliff Lake': 159, 'Park Ridge': 114, 'Montvale': 90, 'Pearl River': 118, 'Nanuet': 100, 'Peapack': 117, 'Far Hills': 45, 'Bernardsville': 18, 'Basking Ridge': 12, 'Lyons': 76, 'Millington': 88, 'Stirling': 143, 'Gillette': 48, 'Berkeley Heights': 17, 'Murray Hill': 99, 'New Providence': 104, 'Summit': 145, 'Short Hills': 136, 'Millburn': 87, 'Maplewood': 81, 'South Orange': 140, 'Highland Avenue': 61, 'Orange': 112, 'Brick Church': 23, 'Newark Broad Street': 106, 'Dover': 35, 'Denville': 34, 'Mount Tabor': 94, 'Morris Plains': 91, 'Morristown': 92, 'Convent Station': 30, 'Madison': 77, 'Chatham': 27, 'East Orange': 37, 'Mountain Station': 97, 'Pennsauken': 43298, 'Cherry Hill': 28, 'Lindenwold': 71, 'Atco': 9, 'Hammonton': 55, 'Egg Harbor City': 39, 'Absecon': 2, 'Kingsland': 66, 'Lyndhurst': 75, 'Delawanna': 33, 'Passaic': 115, 'Clifton': 29, 'Paterson': 116, 'Hawthorne': 58, 'Glen Rock Main Line': 52, 'Ridgewood': 131, 'Waldwick': 151, 'Allendale': 3, 'Ramsey Main St': 128, 'Ramsey Route 17': 38417, 'Mahwah': 78, 'Long Branch': 74, 'Raritan': 129, 'Garwood': 47, 'Suffern': 144, 'Atlantic City Rail Terminal': 10, 'Bay Street': 14, 'Glen Ridge': 50, 'Bloomfield': 19, 'Watsessing Avenue': 154, 'Spring Valley': 142, 'Elberon': 40, 'Allenhurst': 4, 'Asbury Park': 8, 'Bradley Beach': 22, 'Belmar': 15, 'Spring Lake': 141, 'Manasquan': 79, 'Point Pleasant Beach': 122, 'Bay Head': 13, 'Gladstone': 49, 'Rutherford': 134, 'Wesmont': 43599, 'Garfield': 46, 'Plauderville': 121, 'Broadway Fair Lawn': 25, 'Radburn Fair Lawn': 126, 'Glen Rock Boro Hall': 51, 'Lake Hopatcong': 67, 'Mount Arlington': 39472, 'Mountain Lakes': 96, 'Boonton': 20, 'Towaco': 147, 'Lincoln Park': 69, 'Mountain View': 98, 'Wayne-Route 23': 39635, 'Little Falls': 72, 'Montclair State U': 38081, 'Montclair Heights': 89, 'Mountain Avenue': 95, 'Upper Montclair': 150, 'Watchung Avenue': 153, 'Walnut Street': 152, 'Hackettstown': 54, 'Mount Olive': 93, 'Netcong': 101, 'High Bridge': 60, 'Annandale': 6, 'Lebanon': 68, 'White House': 157, 'North Branch': 108, 'Port Jervis': 123, 'Otisville': 113, 'Middletown NY': 86, 'Campbell Hall': 26, 'Salisbury Mills-Cornwall': 135, 'Harriman': 57, 'Tuxedo': 149, 'Sloatsburg': 137, 'Jersey Avenue': 32906}

# User Input Form
st.write("## Input Delay Prediction Parameters")

# Time input
time_input = st.time_input("Select Time", step=3600)
hour_of_day = time_input.hour

# From Station Name with search
from_station = st.selectbox(
    "From Station",
    options=list(stations.keys()),
    key="from_station"
)
from_id = stations.get(from_station)

# To Station Name with search
to_station = st.selectbox(
    "To Station",
    options=list(stations.keys()),
    key="to_station"
)
to_id = stations.get(to_station)

# Validate that from and to stations are not the same
if from_station == to_station:
    st.error("The 'From Station' and 'To Station' cannot be the same. Please select different stations.")

# Day of the Week
day_of_week = st.selectbox("Day of the Week", 
                           options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

# Load the local model
model_path = os.path.join('models', 'delay_prediction_model.joblib')
model = joblib.load(model_path)

# Function to map day of week to number
def day_to_number(day):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return days.index(day)

# Prediction function
def predict_delay(hour_of_day, day_of_week, from_id, to_id):
    # Create a feature vector
    input_data = pd.DataFrame([{
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'from_id': from_id,
        'to_id': to_id
    }])
    
    # Predict delay
    predicted_delay = model.predict(input_data)[0]
    return predicted_delay

# Convert day of week to number
day_number = day_to_number(day_of_week)

# Make prediction
if st.button('Predict Delay'):
    try:
        predicted_delay = predict_delay(hour_of_day, day_number, from_id, to_id)

        # Display prediction with larger font and color
        st.write("## Predicted Delay")
        st.markdown(
            f"<h1 style='text-align: center; color: #1E90FF;'>{predicted_delay:.2f} minutes</h1>", 
            unsafe_allow_html=True
        )

        # Use columns for a more structured layout
        col1, col2, col3 = st.columns([1,3,1])
        with col2:
            # Provide context based on delay duration
            if predicted_delay < 5:
                st.success("Your train is likely to be on time or only slightly delayed.")
            elif predicted_delay < 15:
                st.warning("Consider allowing a little extra time for your journey.")
            else:
                st.error("Significant delay expected. Please plan accordingly.")

    except Exception as e:
        st.error(f"Error calculating delay: {str(e)}")

# Display collected inputs in a clean format
st.write("### Input Summary")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Hour of the Day:** {time_input}")
    st.write(f"**From Station:** {from_station} (ID: {from_id})")
with col2:
    st.write(f"**To Station:** {to_station} (ID: {to_id})")
    st.write(f"**Day of the Week:** {day_of_week}")

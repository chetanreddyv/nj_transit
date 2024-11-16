# Import required libraries
import streamlit as st  # For creating web app interface
from PIL import Image  # For handling images
import pandas as pd    # For data manipulation
import numpy as np     # For numerical operations
import joblib         # For loading saved ML models
import os             # For handling file paths

# Define paths to model files
MODEL_DIR = '/Users/chetan/Documents/GitHub/nj_transit/models'
model_path = os.path.join(MODEL_DIR, 'delay_predictor.joblib')      # ML model file
features_path = os.path.join(MODEL_DIR, 'features_list.joblib')     # List of features used in model
metrics_path = os.path.join(MODEL_DIR, 'metrics.joblib')            # Model performance metrics

# Load saved files
model = joblib.load(model_path)                # Load the trained ML model
features_list = joblib.load(features_path)     # Load list of features used during training
metrics = joblib.load(metrics_path)            # Load model performance metrics

# Configure Streamlit page settings
st.set_page_config(
    page_title="NJ Transit Rail Delay Prediction",
    page_icon="🚆",
    layout="wide"    # Use full width of browser
)

# Add custom CSS for responsive title styling
st.markdown("""
    <style>
    .responsive-title {
        font-size: calc(1.5rem + 1.5vw);  # Dynamic font size based on viewport
        font-weight: bold;
        line-height: 1.2;
        padding-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and display logo
logo_path = "assets/new_jesry_transit_logo.png"
logo = Image.open(logo_path)

# Create two-column layout for header
col1, col2 = st.columns([1, 3])  # First column 1/4 width, second column 3/4 width

with col1:
    st.image(logo, use_container_width=True)  # Display logo, scaling to column width

with col2:
    # Display app title with custom styling
    st.markdown('<p class="responsive-title">NJ Transit Rail Delay Prediction 🚆</p>', 
                unsafe_allow_html=True)

# Display model performance metrics in sidebar
st.sidebar.write("### Model Performance Metrics")
for metric_name, value in metrics.items():
    st.sidebar.metric(metric_name, f"{value:.2f}")  # Show each metric with 2 decimal places
    
# Dictionary mapping station names to their IDs
stations = {'Newark Penn Station': 107, 'Union': 38105, 'Roselle Park': 31, 'Cranford': 32, 'Westfield': 155, 'Fanwood': 44, 'Netherwood': 102, 'Plainfield': 120, 'Dunellen': 36, 'Bound Brook': 21, 'Bridgewater': 24, 'Somerville': 138, 'New York Penn Station': 105, 'Secaucus Upper Lvl': 38187, 'Newark Airport': 37953, 'Elizabeth': 41, 'Linden': 70, 'Rahway': 127, 'Metropark': 83, 'Metuchen': 84, 'Edison': 38, 'New Brunswick': 103, 'Princeton Junction': 125, 'Hamilton': 32905, 'Philadelphia': 1, 'Trenton': 148, 'Princeton': 124, 'North Elizabeth': 109, 'Avenel': 11, 'Woodbridge': 158, 'Perth Amboy': 119, 'South Amboy': 139, 'Aberdeen-Matawan': 37169, 'Hazlet': 59, 'Middletown NJ': 85, 'Red Bank': 130, 'Little Silver': 73, 'Hoboken': 63, 'Secaucus Lower Lvl': 38174, 'Wood Ridge': 160, 'Teterboro': 146, 'Essex Street': 43, 'Anderson Street': 5, 'New Bridge Landing': 110, 'River Edge': 132, 'Oradell': 111, 'Emerson': 42, 'Westwood': 156, 'Hillsdale': 62, 'Woodcliff Lake': 159, 'Park Ridge': 114, 'Montvale': 90, 'Pearl River': 118, 'Nanuet': 100, 'Peapack': 117, 'Far Hills': 45, 'Bernardsville': 18, 'Basking Ridge': 12, 'Lyons': 76, 'Millington': 88, 'Stirling': 143, 'Gillette': 48, 'Berkeley Heights': 17, 'Murray Hill': 99, 'New Providence': 104, 'Summit': 145, 'Short Hills': 136, 'Millburn': 87, 'Maplewood': 81, 'South Orange': 140, 'Highland Avenue': 61, 'Orange': 112, 'Brick Church': 23, 'Newark Broad Street': 106, 'Dover': 35, 'Denville': 34, 'Mount Tabor': 94, 'Morris Plains': 91, 'Morristown': 92, 'Convent Station': 30, 'Madison': 77, 'Chatham': 27, 'East Orange': 37, 'Mountain Station': 97, 'Pennsauken': 43298, 'Cherry Hill': 28, 'Lindenwold': 71, 'Atco': 9, 'Hammonton': 55, 'Egg Harbor City': 39, 'Absecon': 2, 'Kingsland': 66, 'Lyndhurst': 75, 'Delawanna': 33, 'Passaic': 115, 'Clifton': 29, 'Paterson': 116, 'Hawthorne': 58, 'Glen Rock Main Line': 52, 'Ridgewood': 131, 'Waldwick': 151, 'Allendale': 3, 'Ramsey Main St': 128, 'Ramsey Route 17': 38417, 'Mahwah': 78, 'Long Branch': 74, 'Raritan': 129, 'Garwood': 47, 'Suffern': 144, 'Atlantic City Rail Terminal': 10, 'Bay Street': 14, 'Glen Ridge': 50, 'Bloomfield': 19, 'Watsessing Avenue': 154, 'Spring Valley': 142, 'Elberon': 40, 'Allenhurst': 4, 'Asbury Park': 8, 'Bradley Beach': 22, 'Belmar': 15, 'Spring Lake': 141, 'Manasquan': 79, 'Point Pleasant Beach': 122, 'Bay Head': 13, 'Gladstone': 49, 'Rutherford': 134, 'Wesmont': 43599, 'Garfield': 46, 'Plauderville': 121, 'Broadway Fair Lawn': 25, 'Radburn Fair Lawn': 126, 'Glen Rock Boro Hall': 51, 'Lake Hopatcong': 67, 'Mount Arlington': 39472, 'Mountain Lakes': 96, 'Boonton': 20, 'Towaco': 147, 'Lincoln Park': 69, 'Mountain View': 98, 'Wayne-Route 23': 39635, 'Little Falls': 72, 'Montclair State U': 38081, 'Montclair Heights': 89, 'Mountain Avenue': 95, 'Upper Montclair': 150, 'Watchung Avenue': 153, 'Walnut Street': 152, 'Hackettstown': 54, 'Mount Olive': 93, 'Netcong': 101, 'High Bridge': 60, 'Annandale': 6, 'Lebanon': 68, 'White House': 157, 'North Branch': 108, 'Port Jervis': 123, 'Otisville': 113, 'Middletown NY': 86, 'Campbell Hall': 26, 'Salisbury Mills-Cornwall': 135, 'Harriman': 57, 'Tuxedo': 149, 'Sloatsburg': 137, 'Jersey Avenue': 32906}

# Create input form section
st.write("## Input Delay Prediction Parameters")

# Time input selector (hourly intervals)
time_input = st.time_input("Select Time", step=3600)
hour_of_day = time_input.hour  # Extract hour for prediction

# Station selection dropdowns with search functionality
from_station = st.selectbox(
    "From Station",
    options=list(stations.keys()),
    key="from_station"
)
from_id = stations.get(from_station)  # Get station ID from dictionary

to_station = st.selectbox(
    "To Station",
    options=list(stations.keys()),
    key="to_station"
)
to_id = stations.get(to_station)  # Get station ID from dictionary

# Validate station selection
if from_station == to_station:
    st.error("The 'From Station' and 'To Station' cannot be the same. Please select different stations.")

# Day of week selection
day_of_week = st.selectbox("Day of the Week", 
                          options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

# Convert day name to number (0=Monday, 6=Sunday)
def day_to_number(day):
    return {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
            "Friday": 4, "Saturday": 5, "Sunday": 6}[day]

def preprocess_features(hour, day, from_id, to_id):
    """Prepare features for model prediction"""
    
    # Create dictionary with all required model features
    features = {
        # Basic time and station features
        'hour_of_day': hour,
        'day_of_week': day,
        'from_id': from_id,
        'to_id': to_id,
        'month': pd.Timestamp.now().month,
        
        # Derived time-based features
        'is_weekend': 1 if day in [5, 6] else 0,
        'is_rush_hour': 1 if hour in [6, 7, 8, 9, 16, 17, 18, 19] else 0,
        
        # Initialize all rail line features to 0
        'line_Atl. City Line': 0,
        'line_Bergen Co. Line ': 0,
        'line_Gladstone Branch': 0,
        'line_Main Line': 0,
        'line_Montclair-Boonton': 0,
        'line_Morristown Line': 0,
        'line_No Jersey Coast': 0,
        'line_Northeast Corrdr': 0,
        'line_Pascack Valley': 0,
        'line_Princeton Shuttle': 0,
        'line_Raritan Valley': 0,
        
        # Set default type and status
        'type_NJ Transit': 1,
        'status_cancelled': 0,
        'status_departed': 1,
        'status_estimated': 0
    }
    
    # Set appropriate line based on station
    if from_id in [107, 105, 38187]:  # Major stations
        features['line_Northeast Corrdr'] = 1
    elif from_id in [49, 117, 45]:     # Gladstone branch
        features['line_Gladstone Branch'] = 1
    else:
        features['line_Main Line'] = 1  # Default line
        
    return features

def predict_delay(hour, day, from_id, to_id):
    """Generate delay prediction"""
    
    # Prepare features in correct format
    features = preprocess_features(hour, day, from_id, to_id)
    input_data = pd.DataFrame([features])
    
    # Ensure features match training data
    input_data = input_data[features_list]
    
    # Get prediction from model
    return model.predict(input_data)[0]

# Prediction button and results display
if st.button('Predict Delay'):
    try:
        # Get prediction
        predicted_delay = predict_delay(
            hour=hour_of_day,
            day=day_to_number(day_of_week),
            from_id=stations[from_station],
            to_id=stations[to_station]
        )

        # Display prediction result
        st.write("## Predicted Delay")
        st.markdown(
            f"<h1 style='text-align: center; color: #1E90FF;'>{predicted_delay:.2f} minutes</h1>", 
            unsafe_allow_html=True
        )

        # Show delay severity indication
        col1, col2, col3 = st.columns([1,3,1])
        with col2:
            if predicted_delay < 5:
                st.success("Train likely to be on time")
            elif predicted_delay < 15:
                st.warning("Minor delay expected")
            else:
                st.error("Significant delay expected")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Display summary of inputs
st.write("### Input Summary")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Time:** {time_input}")
    st.write(f"**From:** {from_station}")
with col2:
    st.write(f"**To:** {to_station}")
    st.write(f"**Day:** {day_of_week}")

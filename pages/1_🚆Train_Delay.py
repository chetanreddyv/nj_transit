import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from huggingface_hub import hf_hub_download

# Page configuration
st.set_page_config(
    page_title="NJ Transit Rail Delay Prediction",
    page_icon="🚆",
    layout="wide"
)

# Custom CSS to make the title responsive
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

# Load and display the logo with title beside it
logo_path = "assets/new_jesry_transit_logo.png"
logo = Image.open(logo_path)

col1, col2 = st.columns([1, 3])  # Adjust the ratio as needed

with col1:
    st.image(logo, use_column_width=True)  # Makes logo responsive

with col2:
    st.markdown('<p class="responsive-title">NJ Transit Rail Delay Prediction 🚆</p>', unsafe_allow_html=True)


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

# Load the trained model
# Replace the model loading line:
# model = joblib.load('models/delay_prediction_model.joblib')
# with:

def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="vsaladi/nj_transit_delay",  # Your actual Hugging Face repo
            filename="delay_prediction_model.joblib",
            token=st.secrets["HF_TOKEN"]
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model once when the app starts
model = load_model()

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
    predicted_delay = predict_delay(hour_of_day, day_number, from_id, to_id)

    # Display prediction with larger font and color
    st.write("## Predicted Delay")
    st.markdown(f"<h1 style='text-align: center; color: #1E90FF;'>{predicted_delay:.2f} minutes</h1>", unsafe_allow_html=True)

    # Use columns for a more structured layout
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        # Provide some context
        if predicted_delay < 5:
            st.success("Your train is likely to be on time or only slightly delayed.")
        elif predicted_delay < 15:
            st.warning("There might be a minor delay. Consider allowing a little extra time for your journey.")
        else:
            st.error("There could be a significant delay. Please plan accordingly and check for any service updates.")
# Display collected inputs
st.write("### Input Summary")
st.write(f"**Hour of the Day:** {time_input}")
st.write(f"**From Station:** {from_station} (ID: {from_id})")
st.write(f"**To Station:** {to_station} (ID: {to_id})")
st.write(f"**Day of the Week:** {day_of_week}")

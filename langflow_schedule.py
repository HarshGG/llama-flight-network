import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import matplotlib.colors as mcolors
import math
import requests
import json



# LangChain Configuration
BASE_API_URL = "https://api.langflow.astra.datastax.com"
LANGFLOW_ID = "def"
FLOW_ID = "abc"
APPLICATION_TOKEN = "xyz"

# Airport coordinates
airport_coords = {
    "SFO": [37.6213, -122.3790],
    "DFW": [32.8998, -97.0403],
    "IAH": [29.9902, -95.3368],
    "AUS": [30.1975, -97.6664],
    "CLT": [35.2140, -80.9431],
    "BOS": [42.3656, -71.0096]
}

# Initialize `scheduled_pax` in session state if not present
if "scheduled_pax" not in st.session_state:
    st.session_state.scheduled_pax = [
        {"origin": "SFO", "destination": "DFW", "scheduled": 150, "forecasted": 140},
        {"origin": "SFO", "destination": "IAH", "scheduled": 120, "forecasted": 100},
        {"origin": "SFO", "destination": "AUS", "scheduled": 110, "forecasted": 90},
        {"origin": "SFO", "destination": "CLT", "scheduled": 140, "forecasted": 120},
        {"origin": "SFO", "destination": "BOS", "scheduled": 130, "forecasted": 110},
        {"origin": "DFW", "destination": "SFO", "scheduled": 140, "forecasted": 120},
        {"origin": "DFW", "destination": "IAH", "scheduled": 200, "forecasted": 180},
        {"origin": "DFW", "destination": "AUS", "scheduled": 150, "forecasted": 130},
        {"origin": "DFW", "destination": "CLT", "scheduled": 170, "forecasted": 150},
        {"origin": "DFW", "destination": "BOS", "scheduled": 160, "forecasted": 140},
        {"origin": "IAH", "destination": "SFO", "scheduled": 110, "forecasted": 90},
        {"origin": "IAH", "destination": "DFW", "scheduled": 180, "forecasted": 160},
        {"origin": "IAH", "destination": "AUS", "scheduled": 130, "forecasted": 110},
        {"origin": "IAH", "destination": "CLT", "scheduled": 140, "forecasted": 120},
        {"origin": "IAH", "destination": "BOS", "scheduled": 150, "forecasted": 130},
        {"origin": "AUS", "destination": "SFO", "scheduled": 110, "forecasted": 90},
        {"origin": "AUS", "destination": "DFW", "scheduled": 130, "forecasted": 110},
        {"origin": "AUS", "destination": "IAH", "scheduled": 120, "forecasted": 100},
        {"origin": "AUS", "destination": "CLT", "scheduled": 110, "forecasted": 90},
        {"origin": "AUS", "destination": "BOS", "scheduled": 140, "forecasted": 120},
        {"origin": "CLT", "destination": "SFO", "scheduled": 140, "forecasted": 120},
        {"origin": "CLT", "destination": "DFW", "scheduled": 170, "forecasted": 150},
        {"origin": "CLT", "destination": "IAH", "scheduled": 140, "forecasted": 120},
        {"origin": "CLT", "destination": "AUS", "scheduled": 130, "forecasted": 110},
        {"origin": "CLT", "destination": "BOS", "scheduled": 150, "forecasted": 130},
        {"origin": "BOS", "destination": "SFO", "scheduled": 130, "forecasted": 110},
        {"origin": "BOS", "destination": "DFW", "scheduled": 160, "forecasted": 140},
        {"origin": "BOS", "destination": "IAH", "scheduled": 150, "forecasted": 130},
        {"origin": "BOS", "destination": "AUS", "scheduled": 140, "forecasted": 120},
        {"origin": "BOS", "destination": "CLT", "scheduled": 170, "forecasted": 150}
    ]


# Direct routes between airports
direct_routes = {
    ("SFO", "DFW"), ("DFW", "SFO"),
    ("DFW", "IAH"), ("IAH", "DFW"),
    ("DFW", "AUS"), ("AUS", "DFW"),
    ("CLT", "DFW"), ("DFW", "CLT"),
    ("CLT", "SFO"), ("SFO", "CLT"),
    ("BOS", "DFW"), ("DFW", "BOS"), 
    ("BOS", "CLT"), ("CLT", "BOS")
}

# Utility: Smooth curved coordinates
def smooth_curved_coordinates(origin_coords, destination_coords, curvature=0.3, num_points=100):
    lat1, lon1 = origin_coords
    lat2, lon2 = destination_coords
    lat_mid = (lat1 + lat2) / 2
    lon_mid = (lon1 + lon2) / 2
    offset_lat = curvature * (lon2 - lon1)
    offset_lon = -curvature * (lat2 - lat1)
    control_lat = lat_mid + offset_lat
    control_lon = lon_mid + offset_lon
    t_values = np.linspace(0, 1, num_points)
    curve_points = [[
        (1 - t) ** 2 * lat1 + 2 * (1 - t) * t * control_lat + t ** 2 * lat2,
        (1 - t) ** 2 * lon1 + 2 * (1 - t) * t * control_lon + t ** 2 * lon2
    ] for t in t_values]
    return curve_points

# Utility: Offset coordinates to separate overlapping lines
def offset_coordinates(origin_coords, destination_coords, offset_magnitude=0.05):
    lat1, lon1 = origin_coords
    lat2, lon2 = destination_coords
    dx = lat2 - lat1
    dy = lon2 - lon1
    length = math.sqrt(dx ** 2 + dy ** 2)
    offset_x = -dy / length * offset_magnitude
    offset_y = dx / length * offset_magnitude
    new_origin = [lat1 + offset_x, lon1 + offset_y]
    new_destination = [lat2 + offset_x, lon2 + offset_y]
    return new_origin, new_destination

# Utility: Map success rate to color
def success_rate_to_color(success_rate):
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
    return mcolors.to_hex(cmap(success_rate / 100))

# Function to run LangChain flow
def run_flow(message: str) -> dict:
    api_url = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/run/{FLOW_ID}"
    payload = {"input_value": message, "output_type": "chat", "input_type": "chat"}
    headers = {"Authorization": "Bearer " + APPLICATION_TOKEN, "Content-Type": "application/json"}
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

# Streamlit app
st.title("Hub-and-Spoke Flight Network Visualization")

# User input for LangChain
user_message = st.text_input("Enter your query for the flight schedule:")
if st.button("Submit"):
    if user_message:
        langchain_response = run_flow(user_message)
        json_text = str(langchain_response)
        try:
        # Navigate to the response text
            message_text = langchain_response.get("outputs", [{}])[0].get("outputs", [{}])[0].get("results", {}).get("message", {}).get("text", "")
            
            print('nit even in here')
            # Search for the first JSON block enclosed in ```json ... ```
            if "```json" in message_text:
                start_idx = message_text.index("```json") + len("```json")
                end_idx = message_text.index("```", start_idx)
                json_text = message_text[start_idx:end_idx].strip()
                parsed_json = json.loads(json_text)

                # Convert to list if it is a dictionary with numbered keys
                if isinstance(parsed_json, dict):
                    st.session_state.scheduled_pax = list(parsed_json.values())
                    print('converting to list')
                elif isinstance(parsed_json, list):
                    st.session_state.scheduled_pax = parsed_json  # It's already a list
                    print('already list')
                else:
                    st.session_state.scheduled_pax = []
                    st.error("Unexpected JSON structure.")

        except (KeyError, ValueError, json.JSONDecodeError) as e:
            json_text = f"Error extracting JSON: {e}"
            print('csome error')
        
        with open('langflow_output.txt', 'w') as f:
            f.write(json_text)

# Toggle for curved lines
use_curved_lines = st.checkbox("Enable Curved Lines for Direct Flights", value=True)

# Create the map
flight_map = folium.Map(location=[35.0, -97.0], zoom_start=5)
for route in st.session_state.scheduled_pax:
    origin = route["origin"]
    destination = route["destination"]
    scheduled = route["scheduled"]
    forecasted = route["forecasted"]
    success_rate = (forecasted / scheduled) * 100 if scheduled > 0 else 0
    origin_coords = airport_coords.get(origin)
    destination_coords = airport_coords.get(destination)
    if origin_coords and destination_coords:
        # Offset overlapping lines
        offset_origin, offset_destination = offset_coordinates(origin_coords, destination_coords)
        # Add curved lines for direct routes
        if use_curved_lines and (origin, destination) in direct_routes:
            folium.PolyLine(
                smooth_curved_coordinates(origin_coords, destination_coords),
                color=success_rate_to_color(success_rate),
                weight=2,
                tooltip=f"Direct: {origin} -> {destination}\nScheduled: {scheduled}, Forecasted: {forecasted}"
            ).add_to(flight_map)
        folium.PolyLine(
            [offset_origin, offset_destination],
            color=success_rate_to_color(success_rate),
            weight=2,
            dash_array="5, 5",
            tooltip=f"Connecting: {origin} -> {destination}\nScheduled: {scheduled}, Forecasted: {forecasted}"
        ).add_to(flight_map)

# Add airport markers
for airport, coords in airport_coords.items():
    folium.Marker(
        coords,
        tooltip=airport,
        icon=folium.Icon(color="blue" if airport not in {"DFW", "CLT"} else "red")
    ).add_to(flight_map)

# Render the map
st.subheader("Flight Network Visualization")
st_folium(flight_map, width=800, height=600)

from gurobipy import Model, GRB, quicksum
import pandas as pd
import numpy as np
import json
import googlemaps
import datetime
import time as tm
import math
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import requests
import streamlit as st
import uuid  # Import UUID module
import random
import os
from scipy.interpolate import interp1d


# Custom styling
st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .box {
        border-radius: 10px;
        padding: 10px;
        background-color: #f4f4f4;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Apply Custom Styling
st.markdown("""
<style>
.result-box {
    background-color: #f4f4f4;
    padding: 15px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
    line-height: 1.6;
    width: 120%;  /* Keeping width 120% as per your request */
    max-width: 1200px;  /* Ensures it doesn’t go too wide on big screens */
    margin-left: -10%;  /* Shifts it left to keep it centered */
    text-align: left;  /* Keeps text alignment clean */
    display: block;
}
.result-box p {
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.result-box-final {
    background-color: #f4f4f4;
    padding: 15px;
    border-radius: 10px;
    font-size: 22px;
    font-weight: bold;
    line-height: 1.6;
    width: 100%;  /* Keeping width 120% as per your request */
    max-width: 1200px;  /* Ensures it doesn’t go too wide on big screens */
    margin-left: 0%;  /* Shifts it left to keep it centered */
    text-align: left;  /* Keeps text alignment clean */
    display: block;
}
.result-box p {
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "trip_active" not in st.session_state:
    st.session_state.trip_active = False  # Tracks if trip is ongoing
if "current_poi" not in st.session_state:
    st.session_state.current_poi = None  # Tracks current POI
if "visited_pois" not in st.session_state:
    st.session_state.visited_pois = []  # Stores visited POIs
if "remaining_time_budget" not in st.session_state:
    st.session_state.remaining_time_budget = None  # Remaining time
if "remaining_cost_budget" not in st.session_state:
    st.session_state.remaining_cost_budget = None  # Remaining cost
if "all_itineraries" not in st.session_state:
    st.session_state.all_itineraries = []  # Stores all itineraries
if "visited_edges" not in st.session_state:
    st.session_state.visited_edges = set()
if "edges_tt" not in st.session_state:
    st.session_state.edges_tt = {}
if "edges_mode" not in st.session_state:
    st.session_state.edges_mode = {}
if "arrival_times" not in st.session_state:  # ✅ Fix Missing Attribute
    st.session_state.arrival_times = {}
if "next_poi" not in st.session_state:  # ✅ Fix Missing Attribute
    st.session_state.next_poi = None
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0
if "total_utility" not in st.session_state:
    st.session_state.total_utility = 0
if "total_time" not in st.session_state:
    st.session_state.total_time = 0
if "temp_time" not in st.session_state:
    st.session_state.temp_time = 0
if "temp_cost" not in st.session_state:
    st.session_state.temp_cost = 0
if "end_trip_clicked" not in st.session_state:
    st.session_state.end_trip_clicked = False  # Initialize state
if "final_time" not in st.session_state:
    st.session_state.final_time = 0
if "temp_list" not in st.session_state:
        st.session_state.temp_list = []

# st.write(st.session_state)



# Title
st.title("🚀 Trip Planner")

# Dropdown for City Selection
st.subheader(" Select Your City")
cities = ["Select a City","Budapest", "Delhi", "Osaka", "Glasgow", "Vienna", "Perth", "Edinburgh"]
city = st.selectbox("", cities)

st.markdown(f"<p class='big-font'>📍 City Selected: {city}</p>", unsafe_allow_html=True)

if city!="Select a City":
    # Paths to Excel files
    utility_data_path = "data.xlsx"

    utility = pd.read_excel(utility_data_path, sheet_name=city)
    # cost_data = pd.read_excel(cost_data_path, sheet_name=city)
    
    # Create a dictionary mapping POI to its entrance fee
    fees_dict = utility.set_index("poiID")["fees"].to_dict()

    # Wrap everything (title + table) inside a properly aligned gray box
    st.markdown("""
        <div style='background-color:#f4f4f4; padding:20px; border-radius:10px; 
                    width:100%; margin:auto; text-align:center;'>
            <h3>Interesting Places to Visit in {}</h3>
            <table style='width:100%; border-collapse: collapse; text-align: left;'>
                <thead>
                    <tr style='background-color: #bbb;'>
                        <th style='padding: 10px; border-bottom: 2px solid #000;'>POI ID</th>
                        <th style='padding: 10px; border-bottom: 2px solid #000;'>POI Name</th>
                        <th style='padding: 10px; border-bottom: 2px solid #000;'>Theme</th>
                    </tr>
                </thead>
                <tbody>
                    {}
                </tbody>
            </table>
        </div>
    """.format(
        city,
        ''.join(
            f"<tr><td style='padding: 10px; border-bottom: 1px solid #ccc;'>{row.poiID}</td>"
            f"<td style='padding: 10px; border-bottom: 1px solid #ccc;'>{row.poiName}</td>"
            f"<td style='padding: 10px; border-bottom: 1px solid #ccc;'>{row.theme}</td></tr>"
            for _, row in utility.iterrows()
        )
    ), unsafe_allow_html=True)

    st.markdown("#")

    # ✅ Select the day for planning
    selected_day = st.selectbox("Select the day for your itinerary:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    st.subheader("Select Must-See POIs")
    
    poi_ids = utility['poiID'].tolist()
    must_see_pois = st.multiselect("If you have any preference then choose the must-see POIs:", poi_ids, placeholder="Choose multiple options")


    ordering_constraints=[]
    # Ordering Constraints Checkbox
    st.write("")
    st.subheader("Ordering Constraints")
    # Checkbox with bigger text
    constraints = st.checkbox("I want ordering constraints", help="Check this to add ordering constraints")

    # If user selects the checkbox, show text input for constraints
    if constraints:
        ordering_constraints = st.text_input(
            "Enter the constraints in this format: (a,b),(c,d) and so on",
            placeholder="(1,2),(3,4)"
        )

        # Parsing function to convert string input into list of tuples
        def parse_constraints(input_text):
            try:
                # Extract tuples using regex
                matches = re.findall(r"\((\d+),(\d+)\)", input_text)
                parsed_constraints = [(int(a), int(b)) for a, b in matches]  # Convert to list of tuples
                return parsed_constraints
            except Exception as e:
                return str(e)  # Return error message if parsing fails

        # Process user input
        if ordering_constraints:
            ordering_constraints = parse_constraints(ordering_constraints)
            if not ordering_constraints:
                st.warning("⚠️ Invalid format! Please enter in (a,b) format.")
            # else:
            #     st.warning("⚠️ Invalid format! Please enter in (a,b) format.")

    st.subheader("Category Constraints")
    category_constraints = st.checkbox("I want category constraints", help="Check this to add category constraints")
    category_counts = utility["theme"].value_counts()

    theme_bounds = {}
   
    if category_constraints:
        # Convert your theme column counts into a DataFrame
        category_counts = utility["theme"].value_counts().reset_index()
        category_counts.columns = ["Theme", "Count"]

        # Add two new columns for lower & upper bounds (initially empty or None)
        category_counts["Lower bound"] = 0
        category_counts["Upper bound"] = category_counts["Count"]

        # Display the editable table
        # If you're on an older Streamlit version, replace `st.data_editor` with `st.experimental_data_editor`
        # Make only the last two columns editable using column_config
        edited_df = st.data_editor(
            category_counts,
            column_config={
                "Theme": st.column_config.Column(disabled=True),       # read-only
                "Count": st.column_config.Column(disabled=True),       # read-only
                "Lower bound": st.column_config.Column(disabled=False),# editable
                "Upper bound": st.column_config.Column(disabled=False) # editable
            },
            use_container_width=True
        )

        edited_df["Lower bound"] = edited_df.apply(
            lambda row: row["Lower bound"]
            if 0 <= row["Lower bound"] <= row["Count"]
            else 0,  # or e.g. row["Count"] - 1
            axis=1
        )

        edited_df["Upper bound"] = edited_df.apply(
            lambda row: row["Upper bound"]
            if row["Lower bound"] <= row["Upper bound"] <= row["Count"]
            else row["Count"],  # or e.g. row["Count"] - 1
            axis=1
        )

        col_left, col_mid, col_right = st.columns([1, 5, 1])  # Middle is wider

        with col_mid:
            st.dataframe(edited_df, use_container_width=True)
        
         # Option 1: Using dictionary comprehension + iterrows
        theme_bounds = {
            row["Theme"]: (row["Lower bound"], row["Upper bound"])
            for _, row in edited_df.iterrows()
        }


    # Budget Inputs (Inline with Columns)
    st.subheader(" Travel Budget")
    col1, col2 = st.columns(2)
    with col1:
        time_budget = st.number_input("⏳ Time Budget (in hours)", min_value=0.0, step=0.5, format="%.1f")
    with col2:
        cost_budget = st.number_input("💸 Cost Budget (in INR)", min_value=0.0, step=10.0, format="%.2f")

    # Coordinates Input (Formatted Display)
    st.subheader(" Coordinates")
    col1, col2 = st.columns(2)

    with col1:
        source_lat = st.number_input("📍 Source Latitude", format="%.6f")
        source_lon = st.number_input("📍 Source Longitude", format="%.6f")

    with col2:
        dest_lat = st.number_input("📍 Destination Latitude", format="%.6f")
        dest_lon = st.number_input("📍 Destination Longitude", format="%.6f")

    # Display Selected Data (Styled)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📌 Selected Travel Details")

    st.markdown(
        f"""
        <div class='box'>
        <p><b>City:</b> {city}</p>
        <p><b>Time Budget:</b> {time_budget} Minutes</p>
        <p><b>Cost Budget:</b> {cost_budget} Rupees</p>
        <p><b>Source Coordinates:</b> ({source_lat}, {source_lon})</p>
        <p><b>Destination Coordinates:</b> ({dest_lat}, {dest_lon})</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Extract relevant data
    poi_ids = utility['poiID'].tolist()
    poi_ids = [0] + poi_ids + [poi_ids[-1] + 1]

    # ## APPENDING SOURCE AND DESTINATION ROWS IN UTILITY DATAFRAME
    rows_to_add = [
        {
            'poiID': 0,
            'poiName': 'source',
            'lat': source_lat,
            'long': source_lon,
            'theme': 'hotel',
            'Avg Visiting TIme': 0,
            'Utility Score': 0,
            'fees': 0,
            'opening time': datetime.time(hour=0, minute=0, second=0),
            'closing time': datetime.time(hour=23, minute=59, second=59),
            'Monday': 1,
            'Tuesday': 1,
            'Wednesday': 1,
            'Thursday': 1,
            'Friday': 1,
            'Saturday': 1,
            'Sunday': 1,
        },
        {
            'poiID': poi_ids[-1],
            'poiName': 'destination',
            'lat': dest_lat,
            'long': dest_lon,
            'theme': 'hotel',
            'Avg Visiting TIme': 0,
            'Utility Score': 0,
            'fees': 0,
            'opening time': datetime.time(hour=0, minute=0, second=0),
            'closing time': datetime.time(hour=23, minute=59, second=59),
            'Monday': 1,
            'Tuesday': 1,
            'Wednesday': 1,
            'Thursday': 1,
            'Friday': 1,
            'Saturday': 1,
            'Sunday': 1,
        }
    ]

    # 3. Convert that list into a small DataFrame
    new_rows_df = pd.DataFrame(rows_to_add)
    
    utility = pd.concat([utility, new_rows_df], ignore_index=True)
    utility.sort_values(by = 'poiID', inplace = True)

    # ## CREATING REQUIRED DATA STRUCTURES

    visit_times = dict(zip(utility['poiID'], utility['Avg Visiting TIme']))
    utility_scores = dict(zip(utility['poiID'], utility['Utility Score']))

    # creating dict for storing latitude and longitude of poiID

    poi_lat = dict(zip(utility["poiID"], utility["lat"]))
    poi_long = dict(zip(utility["poiID"], utility["long"]))

    #API_KEY="AIzaSyC_BoH3G-H7OuW9zB_DvMuL9vkQFe_Pq48"

    n = len(poi_ids)
    walking_matrix = np.zeros((n, n))
    taxi_matrix = np.zeros((n, n))


    # def get_travel_times_batch(origins, destinations, mode):
    #     """Fetch travel times in smaller batches to avoid API limits."""
    #     url = f"https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix?key={API_KEY}"
        
    #     headers = {
    #         "Content-Type": "application/json",
    #         "X-Goog-Api-Key": API_KEY,
    #         "X-Goog-FieldMask": "originIndex,destinationIndex,duration,distanceMeters"
    #     }
        
    #     # Split into smaller batches (max 25x25 to stay under 625)
    #     batch_size = 25
    #     for origin_batch in [poi_ids[i:i + batch_size] for i in range(0, len(poi_ids), batch_size)]:
    #         for dest_batch in [poi_ids[i:i + batch_size] for i in range(0, len(poi_ids), batch_size)]:
                
    #             # Prepare API Request
    #             data = {
    #                 "origins": [{"waypoint": {"location": {"latLng": {"latitude": poi_lat[o], "longitude": poi_long[o]}}}} for o in origin_batch],
    #                 "destinations": [{"waypoint": {"location": {"latLng": {"latitude": poi_lat[d], "longitude": poi_long[d]}}}} for d in dest_batch],
    #                 "travelMode": mode
    #             }

    #             response = requests.post(url, json=data, headers=headers)
    #             result = response.json()
                
    #             # print("\n API Response:", result)  # Debugging

    #             # Store results in matrix
    #             try:
    #                 for entry in result:
    #                     i = entry["originIndex"] + poi_ids.index(origin_batch[0])
    #                     j = entry["destinationIndex"] + poi_ids.index(dest_batch[0])
                        
    #                     if "duration" in entry:
    #                         duration_sec = int(entry["duration"].replace("s", ""))  # Remove "s"
    #                         travel_time = duration_sec / 60  # Convert to minutes
    #                         walking_matrix[i, j] = travel_time if mode == "WALK" else walking_matrix[i, j]
    #                         taxi_matrix[i, j] = travel_time if mode == "DRIVE" else taxi_matrix[i, j]
    #             except Exception as e:
    #                 print("ERROR:", e)

    # Extract opening and closing times into dictionaries
    opening_times = utility.set_index("poiID")["opening time"].to_dict()
    closing_times = utility.set_index("poiID")["closing time"].to_dict()

    # Dictionary storing taxi cost per km

    taxi_cost_per_km = {
        "Delhi": 45,
        "Budapest": 90,
        "Vienna": 345,
        "Osaka": 300,
        "Edinburgh": 135,
        "Glasgow": 115,
        "Perth": 100,
        "Toronto": 260
    }

    taxi_speed = 30 #in kmph
    taxi_cost_per_hour = taxi_speed * taxi_cost_per_km[city]
    taxi_cost_per_min = taxi_cost_per_hour/60

    # taxi_cost_per_meter = taxi_cost_per_km[city] / 1000  

    st.markdown("""
        <style>
        div.stButton > button {
            background-color: red;  /* Blue background */
            color: white;               /* White text */
            padding: 10px 20px;         /* Padding for a better look */
            border: none;               /* Remove border */
            border-radius: 8px;         /* Rounded corners */
            font-size: 16px;            /* Increase font size */
            font-weight: bold;          /* Bold text */
            cursor: pointer;            /* Pointer on hover */
            transition: background-color 0.3s ease; /* Smooth transition */
            margin-left:37%;
        }
        div.stButton > button:hover {
            background-color: #faad9c;  /* Darker blue on hover */
        }
        </style>
        """, unsafe_allow_html=True)
    
    
    # Define file paths
    folder_path = "CSV_Files"

    # Extract file names with timestamps in HHMM format
    walking_files = {int(f.split('_')[1].split('.')[0]): os.path.join(folder_path, f) 
                    for f in os.listdir(folder_path) if f.startswith("W")}
    taxi_files = {int(f.split('_')[1].split('.')[0]): os.path.join(folder_path, f) 
                for f in os.listdir(folder_path) if f.startswith("T")}

    # Load all matrices into a dictionary {HHMM: dataframe}
    def load_matrices(file_dict):
        matrices = {}
        for time_hhmm, file in file_dict.items():
            df = pd.read_csv(file, index_col=0)
            df.columns = df.columns.astype(int)
            df.index = df.index.astype(int)
            matrices[time_hhmm] = df
        return matrices

    # Load travel time matrices
    walking_matrices = load_matrices(walking_files)
    taxi_matrices = load_matrices(taxi_files)

    # Function to estimate travel time using interpolation {matrices is a dictionary (HHMM: Dataframe) format at that time}
    def interpolate_travel_time(arrival_time_minutes, matrices):
        poi_ids = list(next(iter(matrices.values())).index)  # Get POI IDs from any matrix
         # Create an empty DataFrame with POI IDs as index and columns
        estimated_travel_df = pd.DataFrame(index=poi_ids, columns=poi_ids)

        for i in poi_ids:
            for j in poi_ids:
                if i != j:
                    # Convert matrix timestamps (HHMM) to minutes
                    times_hhmm = sorted(matrices.keys())  # Sorted HHMM times
                    times_minutes = [hhmm_to_minutes(t) for t in times_hhmm]  # Convert to minutes
                    travel_values = [matrices[t].loc[i, j] for t in times_hhmm]  # Travel time at each timestamp

                    # Interpolate
                    interp_func = interp1d(times_minutes, travel_values, kind="linear", fill_value="extrapolate")
                    estimated_travel_df.loc[i, j] = interp_func(arrival_time_minutes)
                else:
                    estimated_travel_df.loc[i, j]=0

        return estimated_travel_df.astype(float)
    

    # Convert HHMM format to minutes from 00:00
    def hhmm_to_minutes(hhmm):
        return (hhmm // 100) * 60 + (hhmm % 100)
    

    def minutes_to_hhmm(minutes):
        minutes = int(minutes)  # Convert to integer
        return f"{minutes // 60:02d}:{minutes % 60:02d}"



    def run_optimizer(source_poi, remaining_time_budget, remaining_cost_budget):
        """
        Runs the Gurobi optimizer dynamically based on the current source POI and updated budgets.
        """
        st.write(f"✅ Itinerary from {source_poi} with time budget {remaining_time_budget} and cost budget as {remaining_cost_budget}")


        # Create a dictionary for day availability
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_availability = {day: {} for day in days_of_week}

        # Populate the dictionary
        for i, row in utility.iterrows():
            poi_id = row['poiID']
            for day in days_of_week:
                day_availability[day][poi_id] = row[day]


        # ✅ Call API in batches to update travel times
        # get_travel_times_batch(poi_ids, poi_ids, "WALK")
        # get_travel_times_batch(poi_ids, poi_ids, "DRIVE")

        # Load travel time matrices from CSV files
        # walking_df = pd.read_csv("C:/Users/Neelu/Desktop/dynamic_notebooks/CSV_Files/CSV_Files/W_1045.csv", index_col=0)  
        # taxi_df = pd.read_csv("C:/Users/Neelu/Desktop/dynamic_notebooks/CSV_Files/CSV_Files/T_1045.csv", index_col=0)  

        # Example: Compute interpolated matrices for an arrival time in minutes
        if st.session_state.trip_active == False:
            arrival_time_minutes = 600
        else:
            arrival_time_minutes=st.session_state.arrival_times.get(source_poi, 600)
        walking_df = interpolate_travel_time(arrival_time_minutes, walking_matrices)
        taxi_df = interpolate_travel_time(arrival_time_minutes, taxi_matrices)
        
        # walking_df
        
        # taxi_df
        
        # st.write(st.session_state.arrival_times.get(source_poi,600))

        # Ensure index and columns are integers (poi IDs)
        walking_df.columns = walking_df.columns.astype(int)
        taxi_df.columns = taxi_df.columns.astype(int)
        walking_df.index = walking_df.index.astype(int)
        taxi_df.index = taxi_df.index.astype(int)

        # Convert DataFrames into dictionary format {(from, to): travel_time}
        travel_times_walking = {(i, j): walking_df.loc[i, j] for i in walking_df.index for j in walking_df.columns if i != j}
        travel_times_taxi = {(i, j): taxi_df.loc[i, j] for i in taxi_df.index for j in taxi_df.columns if i != j}

        # st.write(f"walking_df")
        # walking_df
        # st.write("taxi_df")
        # taxi_df

        # ✅ Convert DataFrames into dictionary format {(from, to): travel_time}
        # travel_times_walking = {(i, j): walking_df.loc[i, j] for i in walking_df.index for j in walking_df.columns if i != j}
        # travel_times_taxi = {(i, j): taxi_df.loc[i, j] for i in taxi_df.index for j in taxi_df.columns if i != j}

        # ✅ Initialize Gurobi model
        model = Model("ILP_Model_1")

        # ✅ Dynamically determine `starting_poi`
        if len(st.session_state.visited_pois) == 0:
            starting_poi = poi_ids[0]  # First run, start from initial POI
        else:
            starting_poi = source_poi  # Use the last visited POI

        # ✅ Set `ending_poi` as always the last POI
        ending_poi = poi_ids[-1]

        # Decision variables
        y = model.addVars(poi_ids, vtype=GRB.BINARY, name="y")  # If POI is selected
        z = model.addVars(poi_ids, poi_ids, vtype=GRB.BINARY, name="z")  # If traveling between two POIs
        w = model.addVars(poi_ids, poi_ids, vtype=GRB.BINARY, name="w")  # Walking
        x = model.addVars(poi_ids, poi_ids, vtype=GRB.BINARY, name="x")  # Taxi
        arrival_time = model.addVars(poi_ids, vtype=GRB.CONTINUOUS, name="arrival_time")

        # ✅ Restore `start_time`, but update it dynamically
        if len(st.session_state.visited_pois) == 0:
            start_time_value = 600  # First trip starts at 10 AM
        else:
            start_time_value = st.session_state.arrival_times.get(source_poi, 600)  # Use last POI arrival time

        start_time = model.addVar(vtype=GRB.CONTINUOUS, name="start_time")
        model.addConstr(start_time == start_time_value, name="DynamicStartTime")

        # ✅ Restore `p` for ordering constraints
        N = len(poi_ids)
        p = model.addVars(poi_ids, vtype=GRB.CONTINUOUS, lb=2, ub=N, name="p")

        # Objective: Maximize utility score
        model.setObjective(quicksum(utility_scores[i] * y[i] for i in poi_ids if i not in st.session_state.visited_pois and i != source_poi), GRB.MAXIMIZE)

        # Constraints

        ## Ensure only one travel mode is chosen between POIs
        model.addConstrs((z[i, j] == w[i, j] + x[i, j] for i in poi_ids for j in poi_ids if i != j), name="MultimodalChoice")

        ## ✅ Restore `SingleModeLimit` constraint
        model.addConstrs((z[i, j] <= 1 for i in poi_ids for j in poi_ids if i != j), name="SingleModeLimit")

        ## Logical connection between y[i] and z[i, j]

        for i in poi_ids:
            for j in poi_ids:
                if i != j and i not in st.session_state.visited_pois and j not in st.session_state.visited_pois:
                    model.addConstr(z[i, j] <= y[i], name=f"TravelStartsFrom_{i}_{j}")
                    model.addConstr(z[i, j] <= y[j], name=f"TravelEndsAt_{i}_{j}")

        ## Must-see POIs constraint
        for poi in must_see_pois:
            if poi not in st.session_state.visited_pois:
                model.addConstr(y[poi] == 1, name=f"MustSee_{poi}")

        ## ✅ Corrected Time Constraint
        model.addConstr(
            quicksum(travel_times_walking.get((i, j), 0) * w[i, j] 
                    for i in poi_ids for j in poi_ids 
                    if i != j and (i, j) not in st.session_state.visited_edges) +
            
            quicksum(travel_times_taxi.get((i, j), 0) * x[i, j] 
                    for i in poi_ids for j in poi_ids 
                    if i != j and (i, j) not in st.session_state.visited_edges) +
            
            quicksum(visit_times[i] * y[i] 
                    for i in poi_ids if i!=source_poi and i not in {starting_poi, ending_poi} and i not in st.session_state.visited_pois)
            
            <= remaining_time_budget,
            name="TimeConstraint"
        )

        # for i in poi_ids:
        #     for j in poi_ids:
        #         if (i, j) in x:
        #             if x[i, j].X > 0.5 and (i, j) not in st.session_state.visited_edges:
        #                 print(i, j)

        ## ✅ Restore CATEGORY CONSTRAINT
        for theme, (lower_bound, upper_bound) in theme_bounds.items():
            theme_count = quicksum(y[i] for i in poi_ids 
                                if i - 1 < len(utility) and 
                                utility.iloc[i - 1]["theme"] == theme)  # Exclude previously visited POIs

            model.addConstr(theme_count >= lower_bound, name=f"Min_{theme}")
            model.addConstr(theme_count <= upper_bound, name=f"Max_{theme}")

        # ## ORDERING CONSTRAINT
        # List of ordering constraints in the form of (a, b)
        M = N + 10  # A sufficiently large number
        for (a, b) in ordering_constraints:
            if a not in st.session_state.visited_pois and b not in st.session_state.visited_pois:
                model.addConstr(
                    p[a] + 1 <= p[b] + M * (1 - y[a]) + M * (1 - y[b]), 
                    name=f"Ordering_{a}_before_{b}"
                )
            if a not in st.session_state.visited_pois and b in st.session_state.visited_pois:
                model.addConstr(
                    y[a] == 0,name=f"Avoid_{a}_if_{b}_present"
                )
                
        model.addConstr(quicksum(z[source_poi, j] for j in poi_ids if j != source_poi and j not in st.session_state.visited_pois) == 1, name="StartConstraint")
        model.addConstr(quicksum(z[i, ending_poi] for i in poi_ids if i != ending_poi and i not in st.session_state.visited_pois) == 1, name="EndConstraint")
        model.addConstr(quicksum(z[i, source_poi] for i in poi_ids if i != source_poi and i not in st.session_state.visited_pois) == 0, name="NoIncomingToSource")
        model.addConstr(quicksum(z[ending_poi, j] for j in poi_ids if j != ending_poi) == 0, name="NoOutgoingFromEnd")

        # CONNECTIVITY CONSTRAINT

        for k in poi_ids:
            if k not in [source_poi, ending_poi] and k not in st.session_state.visited_pois:  # Exclude previously visited POIs
                model.addConstr(
                    quicksum(z[i, k] for i in poi_ids if i != k and i not in st.session_state.visited_pois) == y[k], 
                    name=f"FlowIn_{k}"
                )
                model.addConstr(
                    quicksum(z[k, j] for j in poi_ids if j != k and j not in st.session_state.visited_pois) == y[k], 
                    name=f"FlowOut_{k}"
                )

        # Cost Budget Constraint

        # Extract valid (i, j) pairs from taxi_df (excluding diagonal where i == j)
        valid_edges = [(i, j) for i in taxi_df.index for j in taxi_df.columns if i != j and (i, j) not in st.session_state.visited_edges]

        # Add constraint: Total travel cost (taxi) + entrance fee cost ≤ remaining cost budget
        model.addConstr(
            quicksum(
                taxi_df.loc[i, j] * taxi_cost_per_min * x[i, j]  # Get travel time from taxi_df
                for i, j in valid_edges  # Ensures only valid (i, j) pairs are used
            ) +
            quicksum(
                fees_dict.get(i, 0) * y[i]  # Use y[i] to ensure entrance fees are counted only if POI is included
                for i in taxi_df.index if i not in st.session_state.visited_pois and i!=source_poi  # Exclude previously visited POIs
            ) <= remaining_cost_budget,  # Use dynamically updated cost budget
            "CostBudgetConstraint"
        )

        # ✅ Convert `opening_times` and `closing_times` to minutes
        def convert_to_minutes(time_obj):
            return time_obj.hour * 60 + time_obj.minute
        

        opening_times_fun = {i: convert_to_minutes(t) for i, t in opening_times.items()}
        closing_times_fun = {i: convert_to_minutes(t) for i, t in closing_times.items()}


        
        ## ✅ Updated arrival time constraints
        model.addConstr(arrival_time[source_poi] == start_time, name="StartTimeAtSource")
        for i in poi_ids:
            for j in poi_ids:
                if i != j and (j, i) not in st.session_state.visited_edges:
                    model.addConstr(
                        # arrival_time[i] >= (
                        #     arrival_time[j] +
                        #     (visit_times[j] if j != source_poi else 0) +  # Add visit_times[j] only if j is not the source
                        #     (travel_times_walking.get((j, i), 0) * w[j, i] + travel_times_taxi.get((j, i), 0) * x[j, i])
                        # ) * z[j, i],
                        # name=f"ArrivalTime_{j}_to_{i}"
                        arrival_time[i] >= (
                            arrival_time[j] + visit_times[j] +
                            (travel_times_walking.get((j, i), 0) * w[j, i] + travel_times_taxi.get((j, i), 0) * x[j, i])
                        ) * z[j, i],
                        name=f"ArrivalTime_{j}_to_{i}"
                    )

        # ✅ Apply constraints only to visited POIs
        for i in poi_ids:
            # if y[i].X > 0:  # ✅ Only apply if POI is selected
            model.addConstr(arrival_time[i] >= opening_times_fun[i], name=f"OpeningTime_{i}")  # Cannot enter before opening
            model.addConstr(arrival_time[i] + visit_times[i] <= closing_times_fun[i], name=f"ClosingTime_{i}")  # Must leave before closing


        
        ## Opening and closing day constraints

        # Add constraints to ensure POIs are only selected if they are open on the chosen day
        for i in poi_ids:
            if day_availability[selected_day][i] == 0:  # POI is closed on the selected day
                model.addConstr(y[i] == 0, name=f"Closed_POI_{i}_on_{selected_day}")

        if str(i) in day_availability[selected_day]:
            if day_availability[selected_day][str(i)] == 0:
                model.addConstr(y[i] == 0, name=f"Closed_POI_{i}_on_{selected_day}")

        # Solve the model
        begin_time = tm.time()  # Record start time
        model.optimize()  # Run Gurobi optimization
        end_time = tm.time()  # Record end time
        optimization_time = end_time - begin_time
        st.write(f"Optimization completed in {optimization_time:.4f} seconds.")

        if model.status != GRB.OPTIMAL or model.objVal == 0:
            st.write("No itinerary can be generated at this time and cost budget!")
            st.session_state.end_trip_clicked = True
            st.rerun()
        
        # if model.objVal == 0:
        #     st.write("Optimal Value 0 aa gaya hai")
        #     # st.session_state.end_trip_clicked = True
        #     # st.rerun()

        # model.computeIIS()
        # model.write("infeasible_constraints.ilp")


        selected_pois = [i for i in poi_ids if y[i].X > 0.5 and i not in st.session_state.visited_pois]
        selected_edges = [(i, j) for i in poi_ids for j in poi_ids if i != j and z[i, j].X > 0.5 and (i,j) not in st.session_state.visited_edges]

        ## Storing arrival time for next run
        if model.status == GRB.OPTIMAL and model.objVal > 0:
            # st.write("Optimization hua hai 1")
            for i in poi_ids:
                if y[i].X > 0.5:  # Only store arrival times for selected POIs
                    st.session_state.arrival_times[i] = arrival_time[i].X  # Save optimized value

        
        # ✅ Adding covered edges and both `i` and `j` to visited edges and visted POIs respectively and plotting graph
        if model.status == GRB.OPTIMAL and model.objVal > 0:
            st.session_state.visited_pois.append(source_poi)
            if source_poi!=0 and source_poi!=ending_poi:
                st.session_state.total_utility = st.session_state.total_utility+utility_scores[source_poi]

            # ✅ Now determine the next POI from the already stored visited POIs
            for j in poi_ids:
                if j != source_poi and z[source_poi, j].X > 0.5:
                    last_edge=(source_poi,j)
                    st.session_state.next_poi = j
                    st.session_state.visited_edges.add((source_poi, j))
                    # Check the mode of travel and update edges_mode
                    
                    if x[source_poi, j].X > 0.5:  # Taxi
                        st.session_state.edges_mode[last_edge] = 1
                    elif w[source_poi, j].X > 0.5:  # Walking
                        st.session_state.edges_mode[last_edge] = 2
                    break

            if st.session_state.next_poi :
                st.session_state.current_poi = st.session_state.next_poi  # ✅ Update source POI for next run

            if st.session_state.visited_pois:
                last_visited_poi = st.session_state.visited_pois[-1]  # Last visited POI

                # if len(st.session_state.visited_edges) == 0:
                #     last_edge = None
                
                # Get the visit time for the last POI and add randomness
                visit_time_estimate = visit_times.get(last_visited_poi, 0)
                visit_time_taken = math.floor(random.uniform(0.85 * visit_time_estimate, 1.15 * visit_time_estimate))
                
                # Get the entrance fee for the last POI
                entrance_fee = fees_dict.get(last_visited_poi, 0)
                
                # Initialize travel time and cost
                travel_time_taken = 0
                travel_cost = 0
                
                # Get travel time & cost if last edge exists
                if last_edge:
                    i, j = last_edge  # Extract last travel path
                
                    # Travel time calculation
                    travel_time_estimate = math.floor(travel_times_walking.get((i, j), 0) * w[i, j].x + 
                                            travel_times_taxi.get((i, j), 0) * x[i, j].x)
                    travel_time_taken = math.floor(random.uniform(0.75 * travel_time_estimate, 1.25 * travel_time_estimate))
                    st.session_state.edges_tt[last_edge] = travel_time_taken

                    # Travel cost calculation (only if taxi was taken)
                    if x[i, j].x > 0.5:  
                        travel_cost = taxi_cost_per_min * travel_time_taken

                    st.session_state.total_time += visit_time_taken
                    st.session_state.final_time = visit_time_taken
                    st.session_state.total_cost += entrance_fee
                    st.session_state.final_cost = entrance_fee
                    st.session_state.temp_list.append(visit_time_taken)
                    # st.write(f"Itna visit time add karra hu iss source_poi {last_visited_poi} se: {visit_time_taken}")
                    st.session_state.temp_time = travel_time_taken
                    st.session_state.temp_cost = travel_cost
                    # st.write(f"Utility of {source_poi}:{fees_dict[source_poi]}")
                    # if source_poi==0 or source_poi==ending_poi:
                    #     st.write(f"Utility of {source_poi}:0")
                    # else:
                    #     st.write(f"Utility of {source_poi}:{utility_scores[source_poi]}")
                    # st.write(f"Visit time at {source_poi}:{visit_time_taken}")
                    # st.write(f"Travel time from {source_poi} to {j}:{travel_time_taken}")
                    # st.write(f"Total Time spent at {source_poi}:{visit_time_taken + travel_time_taken}")
                    # st.write(f"Entrance fees of {source_poi}:{entrance_fee}")
                    # st.write(f"Travel cost from {source_poi} to {j}:{travel_cost}")
                    # st.write(f"Total Cost :{entrance_fee + travel_cost}")
                
                # Calculate new budgets
                st.session_state.new_time_budget = max(0, st.session_state.remaining_time_budget - (visit_time_taken + travel_time_taken))
                st.session_state.new_cost_budget = max(0, st.session_state.remaining_cost_budget - (entrance_fee + travel_cost))

            
            st.subheader("Travel Itinerary Graph")

            G_walking = nx.DiGraph()
            merged_nodes = {}

            poi_locations = dict(zip(utility["poiID"], zip(utility["lat"], utility["long"])))

            # Merge nodes that have the same latitude and longitude
            for i in selected_pois:
                lat, lon = poi_locations[i]  # Assuming poi_locations is a dictionary {poiID: (lat, lon)}
                if (lat, lon) in merged_nodes:
                    merged_nodes[(lat, lon)].append(i)
                else:
                    merged_nodes[(lat, lon)] = [i]

            # Add only nodes and edges from the selected itinerary
            # st.write(selected_edges)
            # st.write(poi_locations)
            for start, end in selected_edges:
                # st.write(start)
                # st.write(end)
                start_pos = poi_locations[start]
                end_pos = poi_locations[end]
                # st.write(start_pos)
                # st.write(end_pos)
                merged_start = merged_nodes[start_pos][0]  # Use one representative node
                merged_end = merged_nodes[end_pos][0]
                # st.write(merged_start)
                # st.write(merged_end)
                if (merged_start, merged_end) in travel_times_walking or (merged_start, merged_end) in travel_times_taxi:
                    time = travel_times_walking[(merged_start, merged_end)] if w[(merged_start, merged_end)].x == 1 else travel_times_taxi[(merged_start, merged_end)]
                    G_walking.add_edge(merged_start, merged_end, weight=round(time, 2))

            # Generate positions for the nodes in the graph
            pos = {merged_nodes[loc][0]: nx.circular_layout(G_walking)[merged_nodes[loc][0]] for loc in merged_nodes}

            # Determine travel mode for each edge
            edge_colors = ["gray" if w[(start, end)].x == 1 else "orange" for start, end in G_walking.edges()] ## G_walking graph ka naam hai jiske edges ki baat karre hain
            edge_widths = [3] * len(G_walking.edges())

            fig, ax = plt.subplots(figsize=(10, 6))
            nx.draw(
                G_walking, pos, with_labels=True, node_color='lightblue',
                edge_color=edge_colors, width=edge_widths, node_size=2000, font_size=10, ax=ax
            )

            edge_labels_walking = nx.get_edge_attributes(G_walking, 'weight')
            nx.draw_networkx_edge_labels(G_walking, pos, edge_labels=edge_labels_walking, font_size=10, ax=ax)

            start_pos = poi_locations[source_poi]  # Modified to dynamically get start coordinates
            end_pos = poi_locations[ending_poi]  # Modified to dynamically get end coordinates

            if start_pos == end_pos:
                nx.draw_networkx_nodes(G_walking, pos, nodelist=[merged_nodes[start_pos][0]], node_color='green', node_size=2000, ax=ax)
            else:
                nx.draw_networkx_nodes(G_walking, pos, nodelist=[merged_nodes[start_pos][0]], node_color='green', node_size=2000, ax=ax)
                nx.draw_networkx_nodes(G_walking, pos, nodelist=[merged_nodes[end_pos][0]], node_color='red', node_size=2000, ax=ax)


            start_patch = mpatches.Patch(color='green', label='Start Node')
            end_patch = mpatches.Patch(color='red', label='End Node')
            legend_handles = [start_patch] if start_pos == end_pos else [start_patch, end_patch]
            ax.legend(handles=legend_handles, loc="upper right", fontsize=10, frameon=True, borderpad=1)

            nx.draw_networkx_labels(G_walking, pos, labels={merged_nodes[start_pos][0]: str(merged_nodes[start_pos][0])}, font_color="white", font_size=12, font_weight="bold", ax=ax)
            if start_pos != end_pos:
                nx.draw_networkx_labels(G_walking, pos, labels={merged_nodes[end_pos][0]: str(merged_nodes[end_pos][0])}, font_color="white", font_size=12, font_weight="bold", ax=ax)

            plt.title("Graph Representing Travel Time by Walking and Taxi")

            st.pyplot(fig)


        
            # if model.status == GRB.OPTIMAL:
            itinerary_html = "<div style='background-color:#f0f0f0; padding:20px; border-radius:10px; width:120%; margin:auto;  margin-left: -10%; '>"
            itinerary_html += "<h3 style='text-align:center; color:#333;'>User-Followable Itinerary with Accurate Arrival Times</h3><br>"

            # source_poi = starting_poi
            current_poi = source_poi
            second_last_poi = None

            # new code starts

            # Initialize an empty list to store the ordered itinerary
            itinerary_list = []

            while current_poi >= 0:
                arrival = arrival_time[current_poi].x
                visit_time = visit_times[current_poi]
                next_poi = None
                travel_time = 0

                # Store the current POI in the list
                itinerary_list.append(current_poi)

                # Find the next POI in the sequence
                for j in poi_ids:
                    if z[current_poi, j].x > 0.5:
                        next_poi = j
                        travel_time = (
                            travel_times_walking.get((current_poi, j), 0) * w[current_poi, j].x +
                            travel_times_taxi.get((current_poi, j), 0) * x[current_poi, j].x
                        )
                        break

                # Check if this is the second-last POI
                if next_poi:
                    mode = "Walking" if w[current_poi, next_poi].x > 0.5 else "Taxi"
                    # itinerary_list.append(("Travel", travel_time, mode))  # Store travel details

                    itinerary_html += (
                        f"<p style='font-size:18px; font-weight:bold; color:#000;'>"
                        f"POI({current_poi}), "
                        f"<span style='color:#922b21;'>Arrival Time: {arrival:.0f} min</span>, "
                        f"Visiting Time: {visit_time} min, "
                        f"<span style='color:#922b21;'>Travel Time to POI({next_poi}): {travel_time:.0f} min</span> "
                        f"via <b>{mode}</b></p>"
                    )
                    second_last_poi = current_poi
                else:
                    break  # Stop after the second-last POI

                # Move to the next POI
                current_poi = next_poi

            # Handle the last POI manually
            if second_last_poi:
                last_poi = current_poi
                last_arrival_time = arrival_time[second_last_poi].x + visit_times[second_last_poi] + (
                    travel_times_walking.get((second_last_poi, last_poi), 0) * w[second_last_poi, last_poi].x +
                    travel_times_taxi.get((second_last_poi, last_poi), 0) * x[second_last_poi, last_poi].x
                )
                # itinerary_list.append(last_poi)  # Store final POI

                itinerary_html += (
                    f"<p style='font-size:18px; font-weight:bold; color:#000;'>"
                    f"POI({last_poi}) "
                    f"<span style='color:#922b21;'>Arrival Time: {last_arrival_time:.0f} min</span>, "
                    f"Visiting Time: {visit_times[last_poi]} min</p>"
                )

            itinerary_html += "</div>"  # Closing the gray box div

            st.markdown(itinerary_html, unsafe_allow_html=True)

            # st.write(itinerary_list)  # You can now use this ordered list for further processing
            st.session_state.all_itineraries.append(itinerary_list)

        
        if model.status == GRB.OPTIMAL and model.objVal>0:
            # st.write("Optimization hua hai 2")
            optimized_utility_score = model.objVal
    
            # Extract selected POIs
            # selected_pois = [i for i in poi_ids if y[i].X > 0.5]

            # Compute total taxi travel cost using taxi_df
            total_taxi_cost = sum(
                taxi_df.loc[i, j] * taxi_cost_per_min * x[i, j].X
                for i, j in valid_edges if x[i, j].X > 0.5 #Include only selected taxi routes,valid edges calc me hi not in st.session_state.visited_edge is considered
            )

            # Compute total entrance fee cost
            total_entrance_fee_cost = sum(
                fees_dict.get(i, 0) for i in taxi_df.index 
                if i not in st.session_state.visited_pois and i != source_poi and y[i].X > 0
            )

            # Compute total trip cost
            total_trip_cost = total_taxi_cost + total_entrance_fee_cost
        else:
            print("No optimal solution found.")

        print()

        if model.status == GRB.OPTIMAL and model.objVal>0:
            total_travel_time = 0  # Initialize total travel time

            for i in poi_ids:
                for j in poi_ids:
                    if i != j and (i, j) in selected_edges:
                        travel_time = (travel_times_walking.get((i, j), 0) * w[i, j].x +
                                    travel_times_taxi.get((i, j), 0) * x[i, j].x)
                        
                        # selected_travel_edges.append((i, j, travel_time))  # Store edge and its time
                        
                        total_travel_time += travel_time  # Add to total time

            total_visit_time = 0  # Initialize total visit time

            
            # selected_visit_pois = []
            for i in selected_pois:
                visit_time = visit_times[i]  # Get visit time of POI
                if i!=source_poi:
                    total_visit_time += visit_time  # Add to total visit time

            total_time_taken = total_travel_time + total_visit_time
            # st.write(y)
            st.markdown(f"""
            <div class="result-box">
                <p>📊 <b>Optimized Utility Score:</b> {optimized_utility_score:.2f}</p>
                <p>📍 <b>Selected POIs:</b> {selected_pois}</p>
                <hr>
                <p>🚕 <b>Total Taxi Cost:</b> ₹{total_taxi_cost:.2f}</p>
                <p>🎟️ <b>Total Entrance Fee Cost:</b> ₹{total_entrance_fee_cost:.2f}</p>
                <p>🛎️ <b>Total Trip Cost:</b> ₹{total_trip_cost:.2f}</p>
                <hr>
                <p>🚶 <b>Total Travel Time:</b> {total_travel_time:.2f} minutes</p>
                <p>🕒 <b>Total Visit Time:</b> {total_visit_time:.2f} minutes</p>
                <p>⏱️ <b>Total Time Taken:</b> {total_time_taken:.2f} minutes</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("#")
            st.markdown(f"""
            <div class="result-box">
                <p><b>The time we suggested to visit POI </b>{last_visited_poi}<b> : </b>{visit_time_estimate} <b>minutes</b></p>
                <p><b>The time taken by you to visit POI </b>{last_visited_poi}<b> : </b>{visit_time_taken} <b>minutes</b></p>
                <hr>
                <p><b>The time we suggested to travel from POI </b>{last_visited_poi}<b> to </b>{st.session_state.current_poi}<b> : </b>{travel_time_estimate} <b>minutes</b></p>
                <p><b>The time taken by you to travel from POI </b>{last_visited_poi}<b> to </b>{st.session_state.current_poi}<b> : </b>{travel_time_taken} <b>minutes</b></p>
            </div>
            """, unsafe_allow_html=True)
            
         


    # First, initialize session state variables if they don't exist
    if "trip_active" not in st.session_state:
        st.session_state.trip_active = False
    if "current_poi" not in st.session_state:
        st.session_state.current_poi = 0
    if "remaining_time_budget" not in st.session_state:
        st.session_state.remaining_time_budget = time_budget
    if "remaining_cost_budget" not in st.session_state:
        st.session_state.remaining_cost_budget = cost_budget
    if "all_itineraries" not in st.session_state:
        st.session_state.all_itineraries = []
    if "iteration_count" not in st.session_state:
        st.session_state.iteration_count = 0
    if "current_itinerary" not in st.session_state:
        st.session_state.current_itinerary = None
    if "generation_requested" not in st.session_state:
        st.session_state.generation_requested = False

    # Initial generation button
    if not st.session_state.trip_active and not st.session_state.end_trip_clicked:
        if st.button("GENERATE ITINERARY", key="initial_generate"):
            st.session_state.generation_requested = True
            st.session_state.trip_active = True
            st.session_state.current_poi = 0
            st.session_state.remaining_time_budget = time_budget
            st.session_state.remaining_cost_budget = cost_budget
            st.session_state.all_itineraries = []
            
            # Call optimizer and store results
            st.session_state.current_itinerary = run_optimizer(st.session_state.current_poi, time_budget, cost_budget)
            
            
            # Add a button to proceed
            if st.button("Continue with this itinerary", key="continue_initial"):
                st.rerun()

    # Show ongoing trip planning UI when trip is active and not currently in generation process
    elif st.session_state.trip_active and not st.session_state.end_trip_clicked:
        # Display the current state
        st.write(f"Next Source POI: {st.session_state.current_poi}")
        st.write(f"The new time budget is: {st.session_state.new_time_budget} ")
        st.write(f"The new cost budget is: {st.session_state.new_cost_budget} ")
        if st.session_state.current_poi == poi_ids[-1]:
            st.success("🎉 Destination reached! Your trip is complete.")
        else:
            generate_button = st.button("Generate Next Segment")
            
            if generate_button:
                st.session_state.remaining_time_budget = st.session_state.new_time_budget
                st.session_state.remaining_cost_budget = st.session_state.new_cost_budget
                st.session_state.iteration_count += 1
                st.session_state.generation_requested = True
                st.session_state.total_time+=st.session_state.temp_time
                st.session_state.temp_list.append(st.session_state.temp_time)
                # st.write(f"Itna travel karne ka time joda hai {st.session_state.current_poi} tak pahuchne ka: {st.session_state.temp_time}. Visit ho gaya hai ab ye {st.session_state.current_poi}")
                st.session_state.total_cost+=st.session_state.temp_cost
                
                # Call optimizer and store results
                st.session_state.current_itinerary = run_optimizer(
                    st.session_state.current_poi, 
                    st.session_state.new_time_budget, 
                    st.session_state.new_cost_budget
                )
                
                # Add a button to proceed
                if st.button("Continue with this itinerary", key=f"continue_{st.session_state.iteration_count}"):
                    st.session_state.generation_requested = False
                    st.rerun()

# Button to end the trip
if st.session_state.trip_active:
    if st.session_state.end_trip_clicked or st.button("End Trip"):
        st.session_state.trip_active = False
        st.write("✅ **Trip Ended!** Here are all your itineraries:")
        if "all_itineraries" in st.session_state and st.session_state.all_itineraries:
            st.write("## All Itineraries")
            
            for idx, itinerary_list in enumerate(st.session_state.all_itineraries):
                st.write(f"### Itinerary {idx+1}")

                # Convert POI sequence into a string with arrows
                itinerary_str = " → ".join(map(str, itinerary_list))

                # Display the itinerary as a markdown-styled text
                st.markdown(f"*****{itinerary_str}*****")

        st.markdown("#")
        # st.write(f"subtracting last edge time {st.session_state.temp_list[-1]}")
        st.session_state.total_time -= st.session_state.temp_list[-1]

        st.subheader("📌Your Final Itinerary")
        st.markdown(f"""
            <div class="result-box-final">
                <p><b>Destination: </b>{st.session_state.visited_pois[-1]}</p>
                <p><b>Total Utility Covered: </b>{st.session_state.total_utility}</p>
                <p><b>Total Time Spent: </b>{st.session_state.total_time}</p>
                <p><b>Total Cost Incurred: </b>{st.session_state.total_cost}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#")

        # st.write(f"Node {st.session_state.visited_pois[-1]} ka visit time {st.session_state.final_time} liya hai")
        # st.write("These times are added in Total Time Spent:")
        # st.session_state.temp_list
        
        st.write("Final Itinerary followed by you was: ")

        # Create a directed graph
        G_final = nx.DiGraph()
        merged_nodes = {}

        poi_locations = dict(zip(utility["poiID"], zip(utility["lat"], utility["long"])))

        # Merge nodes that have the same latitude and longitude
        for i in st.session_state.visited_pois:
            lat, lon = poi_locations[i]
            if (lat, lon) in merged_nodes:
                merged_nodes[(lat, lon)].append(i)
            else:
                merged_nodes[(lat, lon)] = [i]
                
        for start, end in st.session_state.visited_edges:
            if start in st.session_state.visited_pois and end in st.session_state.visited_pois:
                start_pos = poi_locations[start]
                end_pos = poi_locations[end]
                # st.session_state.visited_pois
                # st.session_state.visited_edges
                # st.write("Mai yaha")
                merged_start = merged_nodes[start_pos][0]  # Use one representative node
                merged_end = merged_nodes[end_pos][0]

                if (merged_start, merged_end) in st.session_state.edges_tt:
                    travel_time = st.session_state.edges_tt[(merged_start, merged_end)]
                    G_final.add_edge(merged_start, merged_end, weight=round(travel_time, 2))

        # Generate positions for the nodes in the graph
        pos = nx.circular_layout(G_final)

        # Determine travel mode and visited status for each edge
        edge_colors = []
        edge_widths = []

        for start, end in G_final.edges():
            if (start, end) in st.session_state.visited_edges:
                if st.session_state.edges_mode.get((start, end), 2) == 1:  # Taxi
                    edge_colors.append("orange")
                else:  # Walking
                    edge_colors.append("gray")
                edge_widths.append(3)
            else:
                edge_colors.append("lightgray")  # Unvisited edges in lighter color
                edge_widths.append(1)

        # Plot the graph
        fig, ax = plt.subplots(figsize=(10, 6))
        nx.draw(
            G_final, pos, with_labels=True, node_color='#03f8fc',
            edge_color=edge_colors, width=edge_widths, node_size=2000, font_size=10, ax=ax
        )

        # Add travel time labels
        edge_labels = nx.get_edge_attributes(G_final, 'weight')
        nx.draw_networkx_edge_labels(G_final, pos, edge_labels=edge_labels, font_size=10, ax=ax)

        # Mark visited POIs
        nx.draw_networkx_nodes(G_final, pos, nodelist=st.session_state.visited_pois, node_color='#03f8fc', node_size=2000, ax=ax)

        # Mark start and end POIs
        start_pos = poi_locations[0]
        end_pos = poi_locations[st.session_state.visited_pois[-1]]
        
     

        nx.draw_networkx_nodes(G_final, pos, nodelist=[merged_nodes[start_pos][0]], node_color='green', node_size=2000, ax=ax)
        if start_pos != end_pos:
            nx.draw_networkx_nodes(G_final, pos, nodelist=[merged_nodes[end_pos][0]], node_color='red', node_size=2000, ax=ax)

        # Legend
        start_patch = mpatches.Patch(color='green', label='Start Node')
        end_patch = mpatches.Patch(color='red', label='End Node')
        visited_patch = mpatches.Patch(color='blue', label='Visited POI')
        walk_patch = mpatches.Patch(color='gray', label='Walk (Visited)')
        taxi_patch = mpatches.Patch(color='orange', label='Taxi (Visited)')
        legend_handles = [start_patch, end_patch, visited_patch, walk_patch, taxi_patch]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=10, frameon=True, borderpad=1)

        plt.title("Graph Representing Travel Time by Walking and Taxi")

        st.pyplot(fig)

        st.markdown("#")

        walking_df = interpolate_travel_time(st.session_state.total_time, walking_matrices)
        taxi_df = interpolate_travel_time(st.session_state.total_time, taxi_matrices)

        ttr_dest_walking=st.session_state.total_time + walking_df.loc[st.session_state.visited_pois[-1], poi_ids[-1]]
        ttr_dest_taxi=st.session_state.total_time + taxi_df.loc[st.session_state.visited_pois[-1], poi_ids[-1]]

        ctr_dest_taxi=taxi_df.loc[st.session_state.visited_pois[-1], poi_ids[-1]] * taxi_cost_per_min

        ttr_dest_walking_hhmm=minutes_to_hhmm(ttr_dest_walking)
        ttr_dest_taxi_hhmm=minutes_to_hhmm(ttr_dest_taxi)

        # taxi_df

        # st.write(taxi_df.loc[st.session_state.visited_pois[-1], poi_ids[-1]])

        # st.write(ttr_dest_taxi)

        # st.write(ctr_dest_taxi)

        # st.write(ttr_dest_taxi_hhmm)
 
        st.subheader("📌 BEST CASE SCENARIO")
        st.markdown(f"""
            <div class="result-box-final">
                <p><b>You will reach your destination by walking at: </b>{ttr_dest_walking_hhmm}</p>
                <p><b>You will reach your destination by taxi at: </b>{ttr_dest_taxi_hhmm} <b> by spending extra </b> {ctr_dest_taxi} <b> Rupees</b></p>
            </div>
            """, unsafe_allow_html=True)

      

                
        
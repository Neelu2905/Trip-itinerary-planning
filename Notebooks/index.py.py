from gurobipy import Model, GRB, quicksum
import pandas as pd
import json
import streamlit as st
import math
import datetime
import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import numpy as np

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



# Title
st.title("🚀 Trip Planner")

# Dropdown for City Selection
st.subheader(" Select Your City")
cities = ["Select a City","Budapest", "Delhi", "Osaka", "Toronto", "Glasgow", "Vienna", "Perth", "Edinburgh"]
city = st.selectbox("", cities)

st.markdown(f"<p class='big-font'>📍 City Selected: {city}</p>", unsafe_allow_html=True)

if city!="Select a City":
    # Paths to Excel files
    utility_data_path = "C:/Users/Neelu/Desktop/Thesis/updated_data/Updated Travel Data.xlsx"
    cost_data_path = "C:/Users/Neelu/Desktop/Thesis/updated_data/Cost Data.xlsx"

    utility = pd.read_excel(utility_data_path, sheet_name=city)
    cost_data = pd.read_excel(cost_data_path, sheet_name=city)

# Show interesting places box once city is selected
if city!="Select a City":
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

    st.subheader("Select Must-See POIs")
    
    poi_ids = utility['poiID'].tolist()
    must_see_pois = st.multiselect("If you have any preference then choose the must-see POIs:", poi_ids, placeholder="Choose multiple options")

    ordering_constraints=[]
    # Ordering Constraints 
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

    # Example Haversine function returning distance in meters
    def haversine_distance_meters(lat1, lon1, lat2, lon2):
        """
        Calculate the great-circle distance between two points (lat1, lon1) and
        (lat2, lon2) on Earth in meters.
        """
        # Radius of Earth in meters
        R = 6371000  # ~6,371 km in meters

        # Convert degrees to radians
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)

        # Haversine formula
        a = (math.sin(d_lat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))

        # Distance in meters
        distance_meters = R * c
        return distance_meters



    # Add distance columns (in meters) to the DataFrame
    utility["distance from source"] = utility.apply(
        lambda row: haversine_distance_meters(source_lat, source_lon, row["lat"], row["long"]),
        axis=1
    )

    utility["distance from destination"] = utility.apply(
        lambda row: haversine_distance_meters(dest_lat, dest_lon, row["lat"], row["long"]),
        axis=1
    )

    # Extract relevant data
    poi_ids = [0] + poi_ids + [poi_ids[-1] + 1]

    # ## APPENDING SOURCE COST DATA IN COST_DATA DATAFRAME

    # In[427]:
    new_df = utility[['poiID', 'distance from source']].copy()
    new_df.insert(0,'from',0)
    new_df.rename(columns={'poiID': 'to', 'distance from source': 'cost'}, inplace=True)
    poi_dict = dict(zip(utility["poiID"], utility["theme"]))
    profit_dict = dict(zip(utility["poiID"], utility["Utility Score"]))
    new_df['profit'] = new_df['to'].map(profit_dict)
    new_df['category'] = new_df['to'].map(poi_dict)
    # Step 1: Make a copy of new_df
    reversed_df = new_df.copy()
    # Step 2: Swap the 'from' and 'to' columns
    reversed_df['from'], reversed_df['to'] = reversed_df['to'], reversed_df['from']
    # Step 3: Set profit to 0 and category to 'hotel'
    reversed_df['profit'] = 0
    reversed_df['category'] = 'hotel'
    # Step 4: Append reversed_df to new_df
    new_df = pd.concat([new_df, reversed_df], ignore_index=True)
    # Now new_df contains both (i -> j) and (j -> i) rows.


    # ## APPENDING DESTINATION COST DATA IN COST_DATA DATAFRAME

    new_df_dest = utility[['poiID', 'distance from destination']].copy()
    new_df_dest.insert(0, 'from', poi_ids[-1])
    new_df_dest.rename(columns={'poiID': 'to', 'distance from destination': 'cost'}, inplace=True)
    new_df_dest['profit'] = new_df_dest['to'].map(profit_dict)
    new_df_dest['category'] = new_df_dest['to'].map(poi_dict)
    # Step 1: Make a copy of new_df_dest
    reversed_df_dest = new_df_dest.copy()
    # Step 2: Swap the 'from' and 'to' columns
    reversed_df_dest['from'], reversed_df_dest['to'] = reversed_df_dest['to'], reversed_df_dest['from']
    # Step 3: Set profit to 0 and category to 'hotel'
    reversed_df_dest['profit'] = 0
    reversed_df_dest['category'] = 'hotel'
    # Step 4: Append reversed_df_dest to new_df_dest
    new_df_dest = pd.concat([new_df_dest, reversed_df_dest], ignore_index=True)
    # Now new_df_dest contains both (i -> j) and (j -> i) rows.
    cost_data = pd.concat([cost_data, new_df, new_df_dest], ignore_index=True)


    # ## ADDING SOURCE POI TO DESTINATION POI AND VICE VERSA TO COST_DATA
    # 1. Calculate distances using your Haversine function
    distance_0_end_poi = haversine_distance_meters(source_lat, source_lon, dest_lat, dest_lon)
    # 2. Create new rows as a list of dictionaries
    rows_to_add = [
        {
            'from': 0,
            'to': poi_ids[-1],
            'cost': distance_0_end_poi,
            'profit': 0,
            'category': 'hotel'
        },
        {
            'from': poi_ids[-1],
            'to': 0,
            'cost': distance_0_end_poi,
            'profit': 0,
            'category': 'hotel'
        }
    ]
    # 3. Convert that list into a small DataFrame
    new_rows_df = pd.DataFrame(rows_to_add)
    # 4. Concatenate with your existing cost_data
    cost_data = pd.concat([cost_data, new_rows_df], ignore_index=True)
    utility.drop(columns=['distance from source','distance from destination'], inplace = True)
    cost_data.sort_values(by=['from', 'to'], ascending=[True, True], inplace=True)

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
    # 4. Concatenate with your existing cost_data
    utility = pd.concat([utility, new_rows_df], ignore_index=True)
    utility.sort_values(by = 'poiID', inplace = True)

    # st.dataframe(utility)
    # st.dataframe(cost_data)

    # ## CREATING REQUIRED DATA STRUCTURES

    visit_times = dict(zip(utility['poiID'], utility['Avg Visiting TIme']))
    utility_scores = dict(zip(utility['poiID'], utility['Utility Score']))
    poi_lat = dict(zip(utility['poiID'], utility['lat']))
    poi_long = dict(zip(utility['poiID'], utility['long']))

    # st.write(poi_lat)
    # st.write(poi_long)


    # Create a dictionary for travel times in minutes (cost in meters converted to km then multiplied by 15)
    travel_times_walking = {(row['from'], row['to']): (row['cost'] / 1000) * 15 for _, row in cost_data.iterrows()}
    travel_times_taxi = {(row['from'], row['to']): (row['cost'] / 1000) * 2 for _, row in cost_data.iterrows()}

    # Extract opening and closing times into dictionaries
    opening_times = utility.set_index("poiID")["opening time"].to_dict()
    closing_times = utility.set_index("poiID")["closing time"].to_dict()

    # Dictionary storing taxi cost per km

    taxi_cost_per_km = {
        "Delhi": 17,
        "Budapest": 90,
        "Vienna": 345,
        "Osaka": 300,
        "Edinburgh": 135,
        "Glasgow": 115,
        "Perth": 100,
        "Toronto": 260
    }

    taxi_cost_per_meter = taxi_cost_per_km[city] / 1000  

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


    # Button to Generate Output
    if st.button("GENERATE ITINERARY"):
        
        # ## MODEL DECLARATION, VARIABLES & OBJECTIVE FUNCTION
        model = Model("ILP_Model_1")

        # Decision variables: y[i] = 1 if POI i is included in the itinerary, 0 otherwise
        y = model.addVars(poi_ids, vtype=GRB.BINARY, name="y")
        # Introduce new binary variables for travel between POIs
        z = model.addVars(poi_ids, poi_ids, vtype=GRB.BINARY, name="z")
        start_time = model.addVar(vtype=GRB.CONTINUOUS, name="start_time")
        arrival_time = model.addVars(poi_ids, vtype=GRB.CONTINUOUS, name="arrival_time")
        N = len(poi_ids)  # Total number of POIs
        # Create continuous variables for the position of each POI in the sequence
        p = model.addVars(poi_ids, vtype=GRB.CONTINUOUS, lb=2, ub=N, name="p")
        w = model.addVars(poi_ids, poi_ids, vtype=GRB.BINARY, name="w")  # Walking
        x = model.addVars(poi_ids, poi_ids, vtype=GRB.BINARY, name="x")  # Taxi
        # Objective: Maximize the sum of utility scores for selected POIs
        model.setObjective(quicksum(utility_scores[i] * y[i] for i in poi_ids), GRB.MAXIMIZE)
        # Ensure only one mode is chosen for travel between i and j
        model.addConstrs((z[i, j] == w[i, j] + x[i, j] for i in poi_ids for j in poi_ids if i != j), name="MultimodalChoice")
        model.addConstrs((z[i, j] <= 1 for i in poi_ids for j in poi_ids if i != j), name="SingleModeLimit")

        # ## LOGICAL CONNECTION BETWEEN Y AND Z

        # Additional constraint: Ensure logical connection between y[i] and z[i, j]
        for i in poi_ids:
            for j in poi_ids:
                if i != j:
                    model.addConstr(z[i, j] <= y[i], name=f"TravelStartsFrom_{i}_{j}")
                    model.addConstr(z[i, j] <= y[j], name=f"TravelEndsAt_{i}_{j}")


        ## MUST SEE POIS CONSTRAINT 
        
        for poi in must_see_pois:
            model.addConstr(y[poi] == 1, name=f"MustSee_{poi}")

        # ## TIME CONSTRAINT

        model.addConstr(
            quicksum(travel_times_walking.get((i, j), 0) * w[i, j] for i in poi_ids for j in poi_ids if i != j) +
            quicksum(travel_times_taxi.get((i, j), 0) * x[i, j] for i in poi_ids for j in poi_ids if i != j) +
            quicksum(visit_times[i] * y[i] for i in poi_ids) <= time_budget,
            name="TimeConstraint"
        )


        # ## CATEGORY CONSTRAINT

        # Define lower and upper bounds for each theme
        # theme_bounds = {
        #     "Palace": (0, 3),
        #     "Historical": (0, 3),
        #     "Museum": (0, 5),
        #     "zoo":(0,5),
        #     "Shopping":(0,3)
        # }

        # st.write(theme_bounds)

        # Add constraints for each theme
        for theme, (lower_bound, upper_bound) in theme_bounds.items():
            # Sum up the binary variables y[i] for all POIs belonging to the current theme
            theme_count = quicksum(y[i] for i in poi_ids if i - 1 < len(utility) and utility.iloc[i - 1]["theme"] == theme)

            
            # Add lower and upper bound constraints
            model.addConstr(theme_count >= lower_bound, name=f"Min_{theme}")
            model.addConstr(theme_count <= upper_bound, name=f"Max_{theme}")


        # ## ORDERING CONSTRAINT
        # List of ordering constraints in the form of (a, b)
        # ordering_constraints = [(3,4),(14,19),(10,13)]  
        
        M = N+10  # A sufficiently large number
        for (a, b) in ordering_constraints:
            model.addConstr(
                p[a] + 1 <= p[b] + M * (1 - y[a]) + M * (1 - y[b]), 
                name=f"Ordering_{a}_before_{b}"
            )

        # ## STARTING AND ENDING CONSTRAINT (ALWAYS INCLUDE THEM IN ITINERARY)
        starting_poi = poi_ids[0]
        ending_poi = poi_ids[-1]

        # Start point: Exactly one outgoing edge from node SOURCE
        model.addConstr(quicksum(z[starting_poi, j] for j in poi_ids if j != starting_poi) == 1, name="StartConstraint")

        # End point: Exactly one incoming edge to node DESTINATION
        model.addConstr(quicksum(z[i, ending_poi] for i in poi_ids if i != ending_poi) == 1, name="EndConstraint")

        # ## CONNECTIVITY CONSTRAINT

        for k in poi_ids:
            if k not in [starting_poi, ending_poi]:  # Skip start and end points
                model.addConstr(quicksum(z[i, k] for i in poi_ids if i != k) == y[k], name=f"FlowIn_{k}")
                model.addConstr(quicksum(z[k, j] for j in poi_ids if j != k) == y[k], name=f"FlowOut_{k}")


        # ## SUBTOUR ELIMINATION CONSTRAINT

        # Add the subtour elimination constraints
        for i in poi_ids:
            for j in poi_ids:
                if i != j and i > starting_poi and j > starting_poi:  # Skip the start and end POIs
                    model.addConstr(
                        p[i] - p[j] + 1 <= (N - 1) * (1 - z[i, j]),
                        name=f"SubtourElimination_{i}_{j}"
                    )


        # ## NO OUTGOING EDGE FROM ENDING POI CONSTRAINT

        # No outgoing edges from POI 39 (end POI)
        model.addConstr(quicksum(z[ending_poi, j] for j in poi_ids if j != ending_poi) == 0, name="NoOutgoingFromEnd")


        # ## COST BUDGET CONSTRAINT

        # Extract valid (i, j) pairs from cost_data
        valid_edges = set(zip(cost_data["from"], cost_data["to"]))

        # Create a dictionary mapping POI to its entrance fee from utility
        fees_dict = utility.set_index("poiID")["fees"].to_dict()

        # Add constraint: Total travel cost (taxi) + entrance fee cost ≤ cost_budget
        model.addConstr(
            quicksum(
                cost_data.loc[(cost_data["from"] == i) & (cost_data["to"] == j), "cost"].values[0] * taxi_cost_per_meter * x[i, j]
                for i, j in valid_edges  # Ensures only valid (i, j) pairs are used
            ) +
            quicksum(
                fees_dict.get(i, 0) * y[i]  # Use y[i] to ensure entrance fees are counted only if POI is included
                for i in poi_ids
            ) <= cost_budget,
            "CostBudgetConstraint"
        )


        # ## OPENING CLOSING TIME CONSTRAINT

        def convert_to_minutes(time_obj):
            return time_obj.hour * 60 + time_obj.minute

        opening_times = {i: convert_to_minutes(t) for i, t in opening_times.items()}
        closing_times = {i: convert_to_minutes(t) for i, t in closing_times.items()}

        model.addConstr(start_time == 600, name="StartAfter10AM")

        model.addConstr(arrival_time[starting_poi] == start_time, name="StartTimeAtSource")

        for i in poi_ids:
            for j in poi_ids:
                if i != j:
                    model.addConstr(
                        arrival_time[i] >= (arrival_time[j] + visit_times[j] + (travel_times_walking.get((j, i), 0) * w[j, i] + travel_times_taxi.get((j, i), 0) * x[j, i])) * z[j, i],
                        name=f"ArrivalTime_{j}_to_{i}"
                    )

        for i in poi_ids:
            model.addConstr(arrival_time[i] >= opening_times[i], name=f"OpeningTime_{i}")
            model.addConstr(arrival_time[i] <= closing_times[i] - visit_times[i], name=f"ClosingTime_{i}")


        # ## OPENING CLOSING DAY CONSTRAINT

        # Create a dictionary for day availability
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_availability = {day: {} for day in days_of_week}

        # Populate the dictionary
        for i, row in utility.iterrows():
            poi_id = row['poiID']
            for day in days_of_week:
                day_availability[day][poi_id] = row[day]

        # Suppose the user selects a day
        selected_day = 'Wednesday'

        # Add constraints to ensure POIs are only selected if they are open on the chosen day
        for i in poi_ids:
            if day_availability[selected_day][i] == 0:  # POI is closed on the selected day
                model.addConstr(y[i] == 0, name=f"Closed_POI_{i}_on_{selected_day}")

        if str(i) in day_availability[selected_day]:
            if day_availability[selected_day][str(i)] == 0:
                model.addConstr(y[i] == 0, name=f"Closed_POI_{i}_on_{selected_day}")

        # ## MODEL OPTIMIZATION STARTS

        start_time = time.time()  # Record start time
        model.optimize()  # Run Gurobi optimization
        end_time = time.time()  # Record end time
        optimization_time = end_time - start_time
        st.write(f"Optimization completed in {optimization_time:.4f} seconds.")

        # ## RESULTS
        if model.status == GRB.OPTIMAL and model.objVal>0:

            selected_pois = [i for i in poi_ids if y[i].X > 0.5]

            selected_edges = [(i, j) for i in poi_ids for j in poi_ids if i != j and z[i, j].X > 0.5]

            

            # Graph Visualization (Replace Dummy Data with Actual Itinerary Data)
            st.subheader("Travel Itinerary Graph")

            # Create graph
            G_walking = nx.DiGraph()  # Walking mode

            # Add only nodes and edges from the selected itinerary
            for start, end in selected_edges:
                if (start, end) in travel_times_walking or (start, end) in travel_times_taxi:
                    time = travel_times_walking[(start, end)] if w[(start, end)].x == 1 else travel_times_taxi[(start, end)]
                    G_walking.add_edge(start, end, weight=round(time, 2))  # Use correct travel time based on mode

            # Generate positions for the graph using circular layout
            pos = nx.circular_layout(G_walking)

            # Determine travel mode for each edge
            edge_colors = ["gray" if w[(start, end)].x == 1 else "orange" for start, end in G_walking.edges()]
            edge_widths = [3] * len(G_walking.edges())  # Increase edge width for visibility

            # Plot the Graph
            fig, ax = plt.subplots(figsize=(10, 6))
            nx.draw(
                G_walking, pos, with_labels=True, node_color='lightblue',
                edge_color=edge_colors, width=edge_widths, node_size=2000, font_size=10, ax=ax
            )

            # Add edge labels (weights)
            edge_labels_walking = nx.get_edge_attributes(G_walking, 'weight')
            nx.draw_networkx_edge_labels(G_walking, pos, edge_labels=edge_labels_walking, font_size=10, ax=ax)

            # Highlight start and end nodes
            nx.draw_networkx_nodes(G_walking, pos, nodelist=[starting_poi], node_color='green', node_size=2000, ax=ax)
            nx.draw_networkx_nodes(G_walking, pos, nodelist=[ending_poi], node_color='red', node_size=2000, ax=ax)

            # Create legend manually
            start_patch = mpatches.Patch(color='green', label='Start Node')
            end_patch = mpatches.Patch(color='red', label='End Node')
            ax.legend(handles=[start_patch, end_patch], loc="upper right", fontsize=10, frameon=True, borderpad=1)

            # Add labels for start and end nodes
            nx.draw_networkx_labels(G_walking, pos, labels={starting_poi: str(starting_poi)}, font_color="white", font_size=12, font_weight="bold", ax=ax)
            nx.draw_networkx_labels(G_walking, pos, labels={ending_poi: str(ending_poi)}, font_color="white", font_size=12, font_weight="bold", ax=ax)

            plt.title("Graph Representing Travel Time by Walking and Taxi")

            # Display the graph in Streamlit
            st.pyplot(fig)


            # # Find max latitude & longitude among selected POIs
            # max_lat = max(poi_lat[poi] for poi in selected_pois)
            # max_long = max(poi_long[poi] for poi in selected_pois)
            # min_lat = min(poi_lat[poi] for poi in selected_pois)
            # min_long = min(poi_long[poi] for poi in selected_pois)

            # # Adjust grid spacing dynamically based on the number of nodes
            # num_pois = len(selected_pois)
            # grid_spacing = 0.015 + (0.005 * (num_pois // 5))  # Adjust this value for better spacing


            # pos = {poi: (poi_long[poi], poi_lat[poi]) for poi in selected_pois}

            # # Ensure source and destination have the EXACT same position
            # pos[starting_poi] = (poi_long[starting_poi], poi_lat[starting_poi])
            # pos[ending_poi] = (poi_long[ending_poi], poi_lat[ending_poi])

            # for index, poi in enumerate(selected_pois):
            #     # Assign new coordinates based on a scaled grid
            #     lat_step = (index // int(np.sqrt(len(selected_pois)))) * grid_spacing
            #     long_step = (index % int(np.sqrt(len(selected_pois)))) * grid_spacing
            #     pos[poi] = (max_long - long_step, max_lat - lat_step)  # Placing POIs evenly

            # # Create Graph
            # G_walking = nx.Graph()

            # # Add POIs as Nodes
            # for poi in selected_pois:
            #     G_walking.add_node(poi, pos=pos[poi])

            # # Add Edges Based on Selected Itinerary
            # for start, end in selected_edges:
            #     G_walking.add_edge(start, end)

            # # Define Colors
            # start_node = selected_pois[0]
            # end_node = selected_pois[-1]
            # node_colors = ["green" if poi == start_node else "red" if poi == end_node else "lightblue" for poi in selected_pois]

            # # Determine travel mode for each edge
            # edge_colors = ["gray" if w[(start, end)].x == 1 else "orange" for start, end in G_walking.edges()]
            # edge_widths = [3] * len(G_walking.edges())

            # # Plot the Graph
            # fig, ax = plt.subplots(figsize=(10, 8))  # Increase height

            # nx.draw(
            #     G_walking, pos, with_labels=True, node_color=node_colors,
            #     edge_color=edge_colors, width=edge_widths, node_size=2000, font_size=10, ax=ax
            # )

            # # Add edge labels (weights)
            # edge_labels_walking = nx.get_edge_attributes(G_walking, 'weight')
            # nx.draw_networkx_edge_labels(G_walking, pos, edge_labels=edge_labels_walking, font_size=10, ax=ax)

            # # Highlight start and end nodes
            # nx.draw_networkx_nodes(G_walking, pos, nodelist=[start_node], node_color="green", node_size=2000, ax=ax)
            # nx.draw_networkx_nodes(G_walking, pos, nodelist=[end_node], node_color="red", node_size=2000, ax=ax)

            # # Create Legend
            # start_patch = mpatches.Patch(color="green", label="Start Node")
            # end_patch = mpatches.Patch(color="red", label="End Node")
            # ax.legend(handles=[start_patch, end_patch], loc="upper right", fontsize=10, frameon=True, borderpad=1)

            # # Add Latitude-Longitude Grid
            # ax.set_xlabel("Longitude", fontsize=12)
            # ax.set_ylabel("Latitude", fontsize=12)
            # ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

            # # Set axis limits dynamically
            # # ax.set_xlim(max_long - (grid_spacing * np.sqrt(len(selected_pois))), max_long + grid_spacing)
            # # ax.set_ylim(max_lat - (grid_spacing * np.sqrt(len(selected_pois))), max_lat + grid_spacing)
            # # Ensure all nodes fit by dynamically adjusting limits
            # ax.set_xlim(min_long - 0.02, max_long + 0.02)  # Add buffer space
            # ax.set_ylim(min_lat - 0.02, max_lat + 0.02)    # Add extra space at the bottom

            # plt.title("Optimized Graph Representation of Travel Itinerary")
            # st.pyplot(fig)
        

            # if model.status == GRB.OPTIMAL:
            itinerary_html = "<div style='background-color:#f0f0f0; padding:20px; border-radius:10px; width:120%; margin:auto;  margin-left: -10%; '>"
            itinerary_html += "<h3 style='text-align:center; color:#333;'>User-Followable Itinerary with Accurate Arrival Times</h3><br>"

            source_poi = starting_poi
            current_poi = source_poi
            second_last_poi = None

            while current_poi >= 0:
                arrival = arrival_time[current_poi].x
                visit_time = visit_times[current_poi]
                next_poi = None
                travel_time = 0

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
                itinerary_html += (
                    f"<p style='font-size:18px; font-weight:bold; color:#000;'>"
                    f"POI({last_poi}) "
                    f"<span style='color:#922b21;'>Arrival Time: {last_arrival_time:.0f} min</span>, "
                    f"Visiting Time: {visit_times[last_poi]} min</p>"
                )

            itinerary_html += "</div>"  # Closing the gray box div

            st.markdown(itinerary_html, unsafe_allow_html=True)
        else:
            st.error("❌ No optimal solution found. Try increasing cost or time budget")

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


        
        
        
        if model.status == GRB.OPTIMAL and model.objVal>0:
            optimized_utility_score = model.objVal
    
            # Extract selected POIs
            # selected_pois = [i for i in poi_ids if y[i].X > 0.5]

            # Compute total taxi travel cost
            total_taxi_cost = sum(
                cost_data.loc[(cost_data["from"] == i) & (cost_data["to"] == j), "cost"].values[0] * taxi_cost_per_meter * x[i, j].X
                for i, j in valid_edges if x[i, j].X > 0.5  # Include only selected taxi routes
            )

            # Compute total entrance fee cost
            total_entrance_fee_cost = sum(
                fees_dict.get(i, 0) for i in selected_pois  # Sum entrance fees for visited POIs
            )

            # Compute total trip cost
            total_trip_cost = total_taxi_cost + total_entrance_fee_cost

            # Print results
            print(f"Optimized Utility Score: {optimized_utility_score}")
            print(f"Selected POIs: {selected_pois}")
            print()
            print(f"Total Taxi Cost: {total_taxi_cost:.2f} rupees")
            print(f"Total Entrance Fee Cost: {total_entrance_fee_cost:.2f} rupees")
            print(f"Total Trip Cost: {total_trip_cost:.2f} rupees")

        else:
            print("No optimal solution found.")

        print()

        if model.status == GRB.OPTIMAL and model.objVal>0:
            # Calculate the LHS of the time constraint after the solution
            total_travel_time = sum(
                                    travel_times_walking.get((i, j), 0) * w[i, j].x +
                                    travel_times_taxi.get((i, j), 0) * x[i, j].x
                                    for i in poi_ids for j in poi_ids if i != j
                                )

            total_visit_time = sum(visit_times[i] * y[i].X for i in poi_ids)
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
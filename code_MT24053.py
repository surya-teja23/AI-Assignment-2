# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.
    
    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    # Create trip_id to route_id mapping
    for _, row in df_trips.iterrows():
        trip_to_route[row['trip_id']] = row['route_id']

    # Map route_id to a list of stops in order of their sequence
    for _, row in df_stop_times.iterrows():
        route_id = trip_to_route[row['trip_id']]
        route_to_stops[route_id].append(row['stop_id'])

    # Ensure each route only has unique stops
    for route_id in route_to_stops:
        route_to_stops[route_id] = list(dict.fromkeys(route_to_stops[route_id]))

    # Count trips per stop
    for _, row in df_stop_times.iterrows():
        stop_trip_count[row['stop_id']] += 1

    # Create fare rules for routes
    for _, row in df_fare_rules.iterrows():
        fare_rules[row['route_id']] = row['fare_id']

    # Merge fare rules and attributes into a single DataFrame
    merged_fare_df = pd.merge(df_fare_rules, df_fare_attributes, on='fare_id')

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """
    route_trip_count = defaultdict(int)
    for trip_id, route_id in trip_to_route.items():
        route_trip_count[route_id] += 1

    sorted_routes = sorted(route_trip_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_routes[:5]

# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """
    sorted_stops = sorted(stop_trip_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_stops[:5]

# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    stop_route_count = defaultdict(set)
    for route_id, stops in route_to_stops.items():
        for stop in stops:
            stop_route_count[stop].add(route_id)

    stop_route_count = {stop: len(routes) for stop, routes in stop_route_count.items()}
    sorted_stops = sorted(stop_route_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_stops[:5]

# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route. 
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    stop_pairs = defaultdict(list)
    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            stop_pairs[(stops[i], stops[i + 1])].append(route_id)

    one_direct_route_pairs = {pair: routes[0] for pair, routes in stop_pairs.items() if len(routes) == 1}
    sorted_pairs = sorted(one_direct_route_pairs.items(), key=lambda x: stop_trip_count[x[0][0]] + stop_trip_count[x[0][1]], reverse=True)
    return sorted_pairs[:5]

# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    G = nx.Graph()
    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            G.add_edge(stops[i], stops[i + 1], route=route_id)

    pos = nx.spring_layout(G)
    edge_trace = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines', line=dict(width=2, color='blue')))

    node_trace = go.Scatter(x=[], y=[], text=[], mode='markers+text', textposition='top center', marker=dict(size=10, color='red'))
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (node,)

    fig = go.Figure(data=edge_trace + [node_trace], layout=go.Layout(showlegend=False))
    fig.show()

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    direct_routes = []
    for route_id, stops in route_to_stops.items():
        if start_stop in stops and end_stop in stops:
            start_index = stops.index(start_stop)
            end_index = stops.index(end_stop)
            if start_index < end_index:
                direct_routes.append(route_id)
    return direct_routes

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')  
def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  # Clear previous terms
    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")  # Confirmation print

    # Define Datalog predicates
    create_kb()  # Populate the knowledge base
    add_route_data(route_to_stops)  # Add route data to Datalog
    
# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    for route_id, stops in route_to_stops.items():
        for stop in stops:
            +RouteHasStop(route_id, stop)
        for i in range(len(stops) - 1):
            +DirectRoute(route_id, stops[i], stops[i + 1])

# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """
    query_result = DirectRoute(X, start, end)
    return sorted([str(route_id) for route_id in query_result[X]])

# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (int): The ID of the route.
              - stop_id (int): The ID of the stop.
    """
    # Implementation of forward chaining logic
    pass  # Placeholder for actual implementation

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (int): The ID of the route.
              - stop_id (int): The ID of the stop.
    """
    # Implementation of backward chaining logic
    pass  # Placeholder for actual implementation

# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (int): The ID of the route.
              - stop_id (int): The ID of the stop.
    """
    # Implementation of PDDL-style planning logic
    pass  # Placeholder for actual implementation

# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    return merged_fare_df[merged_fare_df['price'] <= initial_fare]

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    route_summary = {}
    for _, row in pruned_df.iterrows():
        route_id = row['route_id']
        if route_id not in route_summary:
            route_summary[route_id] = {'min_price': row['price'], 'stops': set(route_to_stops[route_id])}
        else:
            route_summary[route_id]['min_price'] = min(route_summary[route_id]['min_price'], row['price'])
    return route_summary

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    queue = deque([(start_stop_id, [], 0, initial_fare)])  # (current_stop, path, transfers, remaining_fare)
    visited = set()

    while queue:
        current_stop, path, transfers, remaining_fare = queue.popleft()
        if current_stop == end_stop_id:
            return path

        if transfers > max_transfers or remaining_fare < 0:
            continue

        for route_id, route_info in route_summary.items():
            if current_stop in route_info['stops'] and route_info['min_price'] <= remaining_fare:
                for stop in route_info['stops']:
                    if stop != current_stop and (stop, route_id) not in visited:
                        visited.add((stop, route_id))
                        queue.append((stop, path + [(route_id, stop)], transfers + 1, remaining_fare - route_info['min_price']))

    return []

# Initialize the knowledge base
create_kb()

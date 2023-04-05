import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from math import radians, sin, cos, sqrt, atan2


def calculate_shortest_path(demand_df, location_df):
    # Filter demand data for rows on July 25 and select first 20 rows
    demand_df = demand_df[demand_df['Date'] == '25/07/2022'].head(20)

    # Create list of location IDs for the selected demand locations
    demand_location_ids = demand_df['LocationID'].tolist()

    # Create distance matrix between the selected locations
    num_locations = len(demand_location_ids)
    distance_matrix = {}
    for from_node in range(num_locations):
        from_location = location_df[location_df['LocationID'] == demand_location_ids[from_node]]
        distance_matrix[from_node] = {}
        for to_node in range(num_locations):
            to_location = location_df[location_df['LocationID'] == demand_location_ids[to_node]]
            # Calculate distance between two nodes using Haversine formula
            lat1, lon1 = from_location.iloc[0]['lat'], from_location.iloc[0]['lon']
            lat2, lon2 = to_location.iloc[0]['lat'], to_location.iloc[0]['lon']
            R = 6371  # Earth radius in kilometers
            d_lat = radians(lat2 - lat1)
            d_lon = radians(lon2 - lon1)
            a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            distance = R * c * 1000  # Distance in meters
            distance_matrix[from_node][to_node] = distance

    # Create RoutingIndexManager
    manager = pywrapcp.RoutingIndexManager(num_locations, 1,1)

    # Create RoutingModel
    routing = pywrapcp.RoutingModel(manager)

    # Create distance callback function
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])

    # Define cost of each arc
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem
    assignment = routing.SolveWithParameters(search_parameters)

    # Get the route
    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        route.append(node)
        index = assignment.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))

    # Return the route
    return location_df[location_df['LocationID'].isin(demand_location_ids)].iloc[route][['lon', 'lat']]

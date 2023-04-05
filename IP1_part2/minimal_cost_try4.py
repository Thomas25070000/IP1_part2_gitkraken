from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import math


def create_time_matrix_from_csv(csv_file_locations, csv_file_demand, n, x_depot, y_depot):

    # Read the CSV file into a Pandas DataFrame
    demand_df_original = pd.read_csv(csv_file_demand)
    demand_df_original = demand_df_original.drop_duplicates(subset=['LocationID'])
    demand_df_original['Date'] = pd.to_datetime(demand_df_original['Date'], format='%d/%m/%Y')
    filtered_df = demand_df_original[demand_df_original['Date'] == '2022-07-25']
    first_n = filtered_df.iloc[:n]
    locations = pd.read_csv(csv_file_locations)
    locations_df = pd.merge(first_n, locations, on='LocationID')

    # Add the depot to the locations DataFrame
    depot = {'LocationID': 0, 'lon': x_depot, 'lat': y_depot}
    locations_df = pd.concat([pd.DataFrame(depot, index=[0]), locations_df]).reset_index(drop=True)

    # Create an empty matrix to store the travel times
    num_locations = len(locations_df)
    time_matrix = [[0] * num_locations for i in range(num_locations)]

    # Calculate the travel time between all pairs of locations
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                # Get the (latitude, longitude) coordinates of the two locations
                lat1, lon1 = locations_df.iloc[i][['lat', 'lon']]
                lat2, lon2 = locations_df.iloc[j][['lat', 'lon']]

                # Convert the coordinates to radians
                lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

                # Calculate the Haversine distance between the two locations
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                distance = 6371 * c  # Radius of the Earth in km

                # Convert the distance to a travel time
                travel_time = distance / 50 * 60  # Assume an average speed of 50 km/h

                # Store the travel time in the matrix
                time_matrix[i][j] = int(travel_time/10+1)

    return time_matrix

time_matrix = create_time_matrix_from_csv("/Users/thomasvandendorpe/Dropbox/IP1/Input data/Location_2023.csv","/Users/thomasvandendorpe/Dropbox/IP1/Input data/Demand_2023.csv",16,4.78654,50.915158)


# time_matrix = [
#                 [0, 6, 9, 8, 7, 3, 6, 2, 3, 2, 6, 6, 4, 4, 5, 9, 7],
#                 [6, 0, 8, 3, 2, 6, 8, 4, 8, 8, 13, 7, 5, 8, 12, 10, 14],
#                 [9, 8, 0, 11, 10, 6, 3, 9, 5, 8, 4, 15, 14, 13, 9, 18, 9],
#                 [8, 3, 11, 0, 1, 7, 10, 6, 10, 10, 14, 6, 7, 9, 14, 6, 16],
#                 [7, 2, 10, 1, 0, 6, 9, 4, 8, 9, 13, 4, 6, 8, 12, 8, 14],
#                 [3, 6, 6, 7, 6, 0, 2, 3, 2, 2, 7, 9, 7, 7, 6, 12, 8],
#                 [6, 8, 3, 10, 9, 2, 0, 6, 2, 5, 4, 12, 10, 10, 6, 15, 5],
#                 [2, 4, 9, 6, 4, 3, 6, 0, 4, 4, 8, 5, 4, 3, 7, 8, 10],
#                 [3, 8, 5, 10, 8, 2, 2, 4, 0, 3, 4, 9, 8, 7, 3, 13, 6],
#                 [2, 8, 8, 10, 9, 2, 5, 4, 3, 0, 4, 6, 5, 4, 3, 9, 5],
#                 [6, 13, 4, 14, 13, 7, 4, 8, 4, 4, 0, 10, 9, 8, 4, 13, 4],
#                 [6, 7, 15, 6, 4, 9, 12, 5, 9, 6, 10, 0, 1, 3, 7, 3, 10],
#                 [4, 5, 14, 7, 6, 7, 10, 4, 8, 5, 9, 1, 0, 2, 6, 4, 8],
#                 [4, 8, 13, 9, 8, 7, 10, 3, 7, 4, 8, 3, 2, 0, 4, 5, 6],
#                 [5, 12, 9, 14, 12, 6, 6, 7, 3, 3, 4, 7, 6, 4, 0, 9, 2],
#                 [9, 10, 18, 6, 8, 12, 15, 8, 13, 9, 13, 3, 4, 5, 9, 0, 9],
#                 [7, 14, 9, 16, 14, 8, 5, 10, 6, 5, 4, 10, 8, 6, 2, 9, 0],
#               ]



def generate_time_windows(n):
    time_windows = [(0, 60)]
    time_min = 0
    time_step = int(50/n)# depot time window
    for i in range(1, n+1):
        time_windows.append((time_min, time_min+time_step))
        time_min+=time_step
    return time_windows


time_windows = generate_time_windows(16)



# time_windows = [
#                 (0, 60),  # depot
#                 (7, 30),  # 1
#                 (10, 40),  # 2
#                 (16, 50),  # 3
#                 (10, 30),  # 4
#                 (0, 40),  # 5
#                 (5, 60),  # 6
#                 (0, 50),  # 7
#                 (5, 50),  # 8
#                 (0, 30),  # 9
#                 (10, 40),  # 10
#                 (10, 15),  # 11
#                 (0, 5),  # 12
#                 (5, 10),  # 13
#                 (7, 8),  # 14
#                 (10, 15),  # 15
#                 (11, 15),  # 16
#               ]
def generate_demands(csv_file_locations, csv_file_demand, n):
    # Read the CSV files into Pandas DataFrames
    locations_df = pd.read_csv(csv_file_locations)
    demand_df = pd.read_csv(csv_file_demand)
    demand_df['Date'] = pd.to_datetime(demand_df['Date'], format='%d/%m/%Y')

    # Add up volume for all equal location_id on the same day
    demand_df = demand_df.groupby(['LocationID', 'Date'])['Volume_cm3'].sum().reset_index()

    # Filter the demand DataFrame to get the first n deliveries on July 25
    filtered_demand_df = demand_df[demand_df['Date'] == '2022-07-25'].head(n)

    # Merge the demand and locations DataFrames on LocationID
    merged_df = pd.merge(filtered_demand_df, locations_df, on='LocationID')

    # Generate the demand values
    demands = [0]  # demand at the depot is always 0
    for _, row in merged_df.iterrows():
        demands.append(int(row['Volume_cm3']/10*9))

    return demands

demands = generate_demands("/Users/thomasvandendorpe/Dropbox/IP1/Input data/Location_2023.csv","/Users/thomasvandendorpe/Dropbox/IP1/Input data/Demand_2023.csv",16)
demands = [0, 1, 1, 2, 4, 2, 4, 8, 3, 1, 2, 1, 2, 4, 4, 5, 5]


num_vehicles = 4
demand_20_deliveries = int(sum(demands[0:16]))
vehicle_capacities = [demand_20_deliveries, demand_20_deliveries, demand_20_deliveries, demand_20_deliveries]
vehicle_capacities = [15, 15, 15, 15]

depot_index = 0

time_limit_seconds = 10 # time limit for calculation





def create_data_model(time_matrix, time_windows, num_vehicles, demands, vehicle_capacities, depot_index):
    """Stores the data for the problem."""
    data = {}
    data['time_matrix'] = time_matrix
    data['time_windows'] = time_windows
    data['num_vehicles'] = num_vehicles
    data['demands'] = demands
    data['vehicle_capacities'] = vehicle_capacities
    data['depot'] = depot_index
    return data


"""Capacitated Vehicles Routing Problem (CVRP) with Time Windows."""


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    total_distance = 0
    total_load = 0
    time_dimension = routing.GetDimensionOrDie('Time')
    total_time = 0

    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            route_load += data['demands'][node_index]
            plan_output += 'Place {0:>2} Arrive at {2:>2}min Depart at {3:>2}min (Load {1:>2})\n'.format(
                manager.IndexToNode(index), route_load, solution.Min(time_var), solution.Max(time_var))

            previous_index = index
            index = solution.Value(routing.NextVar(index))

        time_var = time_dimension.CumulVar(index)
        total_time += solution.Min(time_var)
        plan_output += "Place {0:>2} Arrive at {2:>2}min \n\n".format(manager.IndexToNode(index), route_load,
                                                                      solution.Min(time_var), solution.Max(time_var))

        # route output
        plan_output += 'Load of the route: {}\n'.format(route_load)
        plan_output += 'Time of the route: {}min\n'.format(solution.Min(time_var))
        plan_output += "--------------------"

        print(plan_output)
        total_load += route_load

    print('Total load of all routes: {}'.format(total_load))
    print('Total time of all routes: {}min'.format(total_time))


def main():
    """Solve the VRP with time windows."""
    # Instantiate the data problem.
    data = create_data_model(time_matrix, time_windows, num_vehicles, demands, vehicle_capacities, depot_index)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Add Time Windows constraint.
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        30,  # allow waiting time
        30,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)

    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

    # Add time window constraints for each vehicle start node.
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0],
                                                data['time_windows'][0][1])

    # Instantiate route start and end times to produce feasible times.
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = time_limit_seconds

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)
    return solution

solution = main()
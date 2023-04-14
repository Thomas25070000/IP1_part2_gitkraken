"""Simple Vehicles Routing Problem (VRP).

   This is a sample using the routing library python wrapper to solve a VRP
   problem.
   A description of the problem can be found here:
   http://en.wikipedia.org/wiki/Vehicle_routing_problem.

   Distances are in meters.
"""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from build_distance_matrix import return_distance_matrix
from build_distance_matrix import create_demand
import os
import pandas as pd
import matplotlib.pyplot as plt



def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()}')
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print('Total distance of all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}'.format(total_load))
    return total_distance


def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = {}
    for p in range(1,len(distance_matrix)):
        data['charging_times'] = [65]
    data['battery_capacity'] = 47.6
    if os.path.isfile(r'C:\Users\maxge\OneDrive\Desktop\ip1_part 2\distance_matrix'):
        # Load the existing distance matrix
        distance_matrix = pd.read_csv(r'C:\Users\maxge\OneDrive\Desktop\ip1_part 2\distance_matrix', index_col=0)
    else:
        # Create the data and compute the distance matrix
        distance_matrix = return_distance_matrix()
        # Save the distance matrix to disk
        distance_matrix = pd.DataFrame(distance_matrix)
        distance_matrix.to_csv(r'C:\Users\maxge\OneDrive\Desktop\ip1_part 2\distance_matrix')
    #distance_matrix, demand = return_distance_matrix_and_demand()
    data_demand = create_demand()
    demand = data_demand['demand']
    distance_matrix = distance_matrix.values.tolist()
    data['distance_matrix'] = distance_matrix
    print(data['distance_matrix'])

    data['depot'] = 0
    data['demands'] = demand
    print(demand)
    cost_per_m = 0.18
    fixed_cost = 100
    list_driving_costs = []
    list_number_drivers = []
    list_fixed_costs = []
    list_total_costs = []
    for alpha in range(5,101,5):


        data['num_vehicles'] = 30
        data['vehicle_capacities'] = [6000]*data['num_vehicles']

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)



        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)


        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Distance constraint.
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            200000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

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

        def CreateBatteryCallback(distance_matrix, battery_capacity, battery_consumption_rate):
            """Create the battery callback."""

            def battery_callback(from_index, to_index, vehicle_id, battery):
                """Return the battery level after traveling between two nodes."""
                distance = distance_matrix[from_index][to_index]
                battery_capacity_meters = battery_capacity * 1000
                battery_consumption = battery_consumption_rate * distance
                new_battery = battery - (battery_consumption / battery_capacity_meters)
                return max(new_battery, 0)

            return battery_callback

        # Define the battery level constraint.
        battery_consumption_rate = 0.0034  # battery consumption rate per meter (kWh/m)
        battery_callback = CreateBatteryCallback(distance_matrix, battery_capacity, battery_consumption_rate)
        min_battery_level = 0.15  # minimum battery level
        battery_dimension_name = 'Battery'
        routing.AddDimension(
            battery_callback,
            0,  # no slack
            80,  # maximum battery level (percentage)
            True,  # start cumul to zero
            battery_dimension_name)
        for vehicle_id in range(num_vehicles):
            battery_var = routing.GetDimensionOrDie(battery_dimension_name).CumulVar(vehicle_id)
            routing.AddVariableMinimizedByFinalizer(battery_var)
            routing.AddDisjunction([manager.NodeToIndex(depot)], 0)
            routing.AddConstraint(
                battery_var >= min_battery_level * routing.GetDimensionOrDie('Distance').CumulVar(index, vehicle_id))

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(1)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        total_driving_cost = 0
        if solution:
            total_distance = print_solution(data, manager, routing, solution)
            total_driving_cost = int(total_distance*cost_per_m)
        list_driving_costs.append(total_driving_cost)
        list_fixed_costs.append(fixed_cost*alpha)
        list_number_drivers.append(alpha)
        list_total_costs.append(fixed_cost*alpha+total_driving_cost)
    # Create a line plot of the data
    plt.plot(list_number_drivers, list_driving_costs, label='Driving cost')
    plt.plot(list_number_drivers, list_fixed_costs, label='Fixed cost')
    plt.plot(list_number_drivers, list_total_costs, label='Total cost')

    # Set axis labels and title
    plt.xlabel('Number of vehicles')
    plt.ylabel('Cost')
    plt.title('Cost vs. number of vehicles')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == '__main__':
    main()
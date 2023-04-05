import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from haversine import haversine_vector, Unit

def calculate_minimal_cost(n, df_geo, df_loc, df_num, max_number_vehicles_charging, c_per_km, c_per_vehicle, max_km_per_vehicle, date):
    # Convert date column to datetime object
    df_num['Date'] = pd.to_datetime(df_num['Date'], format='%d/%m/%Y')

    # Select demand data for specified date and first 100 demand elements
    df_date = df_num.loc[df_num['Date'] == date].iloc[:100]

    # Merge location and demand dataframes to get the demand locations
    df_loc_num = df_loc.merge(df_date.drop_duplicates(subset=['LocationID']), on='LocationID')

    # Select the first 5 charging stations
    df_charging = df_geo.iloc[:5]

    # Concatenate the location dataframes
    df_loc_all = pd.concat([df_loc_num, df_charging])

    # Create a dictionary mapping location IDs to their corresponding indices
    id_to_idx = {loc_id: i for i, loc_id in enumerate(df_loc_all['LocationID'])}

    # Compute the pairwise distances between all locations
    locations = df_loc_all[['lat', 'lon']].to_numpy()
    dist_mat = haversine_vector(locations, locations, Unit.KILOMETERS)

    # Initialize variables
    min_cost = np.inf
    best_route = None

    # Generate all possible permutations of demand points and charging stations
    for perm in itertools.permutations(df_loc_num['LocationID']):
        # Count the number of times the starting charging station appears in the first n demand points
        start_loc_counts = np.array([perm[:n].count(start_loc) for start_loc in df_charging['LocationID']])

        # Find the indices of charging stations that can allow max_number_vehicles_charging to start there
        feasible_charging_stations = np.where(start_loc_counts <= max_number_vehicles_charging)[0]

        # Add the charging station to the beginning and end of the route
        start_locs = df_charging.iloc[feasible_charging_stations]['LocationID'].values.reshape(-1, 1)
        perm = np.array(perm)  # Convert to NumPy array

        perm_reshaped = perm.reshape(-1, 1)
        route = np.concatenate([start_locs, perm_reshaped, start_locs], axis=0)

        # Calculate the total distance and number of vehicles used for each route
        idx_route = np.vectorize(id_to_idx.get)(route)
        idx_route = idx_route.flatten()  # flatten the array to make it 1D

        total_distance = np.sum(dist_mat[idx_route[:-1], idx_route[1:]], axis=0)  # axis=0 to get a 1D array
        num_vehicles_used = np.ceil(total_distance / max_km_per_vehicle)

        # Calculate the total cost for each route
        total_cost = c_per_km * total_distance + c_per_vehicle * num_vehicles_used

        # Find the best route and minimal cost among the feasible charging stations
        best_feasible_route_idx = np.argmin(total_cost)
        if total_cost[best_feasible_route_idx] < min_cost:
            min_cost = total_cost[best_feasible_route_idx]
            best_route = route[best_feasible_route_idx]

    print('second part')
    # Plot the best route
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    # Plot all locations as black dots
    axs.scatter(df_loc["lon"], df_loc["lat"], s=10, color='black')

    # Plot the best route in blue
    for i in range(len(best_route) - 1):
        axs.plot(df_loc.loc[df_loc['LocationID'].isin(best_route[i:i+2])]['lon'],
                 df_loc.loc[df_loc['LocationID'].isin(best_route[i:i+2])]['lat'],
                 color='blue')

    # Plot charging stations with crosses
    axs.scatter(df_geo["lon"], df_geo["lat"], s=50, color='red', marker='x')

    axs.set_xlabel("Longitude")
    axs.set_ylabel("Latitude")
    axs.set_title("Route Map")

    plt.tight_layout()
    plt.show()

    # Return the minimal cost and best route
    return min_cost, best_route

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c * 1000  # Convert to meters
    return distance
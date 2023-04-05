import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans

def calculate_shortest_path(merged_df: pd.DataFrame, date: str, num_trips: int, num_vehicles: int, charging_stations) -> Tuple[float, List[List[int]]]:

    # Filter the merged DataFrame for the selected trips
    charging_stations = charging_stations.head(num_vehicles)
    selected_trips_df = merged_df.loc[merged_df['Date'] == date].head(num_trips)
    locations = selected_trips_df.drop_duplicates(subset=['LocationID'], keep='first')
    locations = pd.concat([locations, charging_stations], ignore_index=True)

    # Merge the locations DataFrame with the charging_stations DataFrame to get 'lon' and 'lat' values for charging stations

    coordinates = locations[['lat', 'lon']].values

    # Use k-means clustering to group the destinations into num_vehicles clusters
    kmeans = KMeans(n_clusters=num_vehicles, random_state=0).fit(coordinates)
    groups = kmeans.labels_

    total_distance = 0
    paths = []
    plt.figure()  # create a new figure for all the paths
    for group_id in range(num_vehicles):
        group = locations.loc[groups == group_id]['LocationID'].tolist()

        # Find the nearest charging station to the starting location of the group
        start_location = locations.loc[locations['LocationID'] == group[0]].iloc[0]
        charging_stations['distance'] = ((charging_stations['lon'] - start_location['lon'])**2 + (charging_stations['lat'] - start_location['lat'])**2)**0.5
        nearest_charging_station = charging_stations.loc[charging_stations['distance'].idxmin()]

        # Add the charging station to the beginning of the group's list of locations
        group.insert(0, nearest_charging_station['LocationID'])

        G = nx.Graph()
        for loc_id in group:
            row = locations.loc[locations['LocationID'] == loc_id].iloc[0]
            G.add_node(loc_id, pos=(row['lon'], row['lat']))
        for i, node1 in locations.loc[locations['LocationID'].isin(group)].iterrows():
            for j, node2 in locations.loc[locations['LocationID'].isin(group)].iterrows():
                if i != j:
                    distance = ((node1['lon'] - node2['lon'])**2 + (node1['lat'] - node2['lat'])**2)**0.5
                    G.add_edge(node1['LocationID'], node2['LocationID'], weight=distance)

        # find the shortest cycle that visits all nodes once and starts and ends at the same node
        start_id = group[0]
        nodes = list(G.nodes())
        nodes.remove(start_id)
        nodes.insert(0, start_id)
        path = nx.approximation.traveling_salesman_problem(G)

        # compute the total distance of the cycle
        group_distance = sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        total_distance += group_distance
        paths.append(path)

        # plot the results
        pos = nx.get_node_attributes(G, 'pos')
        for loc_id in group:
            row = locations.loc[locations['LocationID'] == loc_id].iloc[0]
            node_color = 'r' if loc_id in charging_stations['LocationID'].tolist() else 'b'
            G.add_node(loc_id, pos=(row['lon'], row['lat']), color=node_color)
        nx.draw_networkx_nodes(G, pos, node_color=[G.nodes[n]['color'] for n in G.nodes()])
        #nx.draw_networkx_nodes(G, pos, node_color='b')
        nx.draw_networkx_edges(G, pos,
                               edgelist=[(path[i], path[i + 1]) for i in range(len(path) - 1)] + [(path[-1], path[0])],
                               edge_color=np.random.rand(3, ), width=2)  # assign random color
    plt.title('Shortest Paths')
    plt.show()

    return total_distance, paths

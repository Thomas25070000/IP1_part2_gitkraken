import pandas as pd
import matplotlib.pyplot as plt
from plot_demand import plot_demand_on_date
from minimal_cost import calculate_minimal_cost
from minimal_cost_try3 import calculate_shortest_path
# Load the data
charging_stations = pd.read_csv("/Users/thomasvandendorpe/Dropbox/IP1/Input data/ChargingStations_2023.csv")
location_df = pd.read_csv("/Users/thomasvandendorpe/Dropbox/IP1/Input data/Location_2023.csv")
demand_df = pd.read_csv("/Users/thomasvandendorpe/Dropbox/IP1/Input data/Demand_2023.csv")

merged_df = pd.merge(demand_df, location_df, on='LocationID')
#plot_demand_on_date('25/07/2022',df_geo,df_loc,df_num)
#min_cost, best_route = calculate_minimal_cost(131, df_geo, df_loc, df_num, 1, 0.2, 10, 200,'25/07/2022')


shortest_path = calculate_shortest_path(merged_df,'25/07/2022',200,4,charging_stations)
print(shortest_path)


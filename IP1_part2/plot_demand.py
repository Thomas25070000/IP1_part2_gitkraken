import pandas as pd
import matplotlib.pyplot as plt

def plot_demand_on_date(date, df_geo, df_loc, df_num):


    # Convert date column to datetime object
    df_num['Date'] = pd.to_datetime(df_num['Date'], dayfirst=True)

    # Select demand data for specified date
    df_date = df_num[df_num['Date'] == date]

    # Merge the demand and location dataframes
    df_loc_num = df_loc.merge(df_date[['LocationID']], on='LocationID')

    # Group the demand dataframe by location and count the number of occurrences
    df_counts = df_date.groupby('LocationID').size().reset_index(name='counts')

    # Plot the data
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    # # Plot all locations as black dots
    # axs.scatter(df_loc["lon"], df_loc["lat"], s=10, color='black')

    # Plot locations in demand with red circles
    for loc in df_loc_num.itertuples():
        loc_id = loc.LocationID
        loc_idx = df_loc.index[df_loc['LocationID'] == loc_id].tolist()
        loc_count = df_counts[df_counts['LocationID'] == loc_id]['counts'].values[0]
        axs.scatter(df_loc.loc[loc_idx]['lon'], df_loc.loc[loc_idx]['lat'], s=20 * loc_count, color='red', marker='o')

    # Plot charging stations with crosses
    axs.scatter(df_geo["lon"], df_geo["lat"], s=50, color='blue', marker='x')

    axs.set_xlabel("Longitude")
    axs.set_ylabel("Latitude")
    axs.set_title("Location Map")

    plt.tight_layout()
    plt.show()

import requests
import json
import urllib
import pandas as pd
import ssl


def create_data():
    """Creates the data."""
    data = {}
    data['API_key'] = 'AIzaSyBhR-s4JcaIZOa8eo1zLJf4px1lTt4vMwE'
    first_n = 17

    locations_df = pd.read_csv("/Users/thomasvandendorpe/Dropbox/IP1/Input data/Location_2023.csv")
    locations = locations_df.head(first_n)
    data['lon'] = locations['lon']
    data['lat'] = locations['lat']
    return data


def create_distance_matrix(data):
    latitudes = data["lat"]
    longitudes = data["lon"]
    API_key = data["API_key"]
    # Distance Matrix API only accepts 100 elements per request, so get rows in multiple requests.
    max_elements = 100
    num_latlngs = len(latitudes)  # total number of latitudes and longitudes provided
    # Maximum number of rows that can be computed per request.
    max_rows = max_elements // num_latlngs
    # num_latlngs = q * max_rows + r
    q, r = divmod(num_latlngs, max_rows)
    latlngs = [f"{lat},{lng}" for lat, lng in zip(latitudes, longitudes)]
    distance_matrix = []
    # Send q requests, returning max_rows rows per request.
    for i in range(q):
        origin_latlngs = latlngs[i * max_rows: (i + 1) * max_rows]
        response = send_request(origin_latlngs, latlngs, API_key)
        distance_matrix += build_distance_matrix(response)

    # Get the remaining remaining r rows, if necessary.
    if r > 0:
        origin_latlngs = latlngs[q * max_rows: q * max_rows + r]
        response = send_request(origin_latlngs, latlngs, API_key)
        distance_matrix += build_distance_matrix(response)
    return distance_matrix

def send_request(origin_latlngs, dest_latlngs, API_key):
    """ Build and send request for the given origin and destination latitudes and longitudes."""
    def build_latlng_str(latlngs):
        # Build a pipe-separated string of latitudes and longitudes
        latlng_str = ''
        for i in range(len(latlngs) - 1):
            latlng_str += latlngs[i] + '|'
        latlng_str += latlngs[-1]
        return latlng_str

    request = 'https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial'
    origin_latlng_str = build_latlng_str(origin_latlngs)
    dest_latlng_str = build_latlng_str(dest_latlngs)
    request = request + '&origins=' + origin_latlng_str + '&destinations=' + \
        dest_latlng_str + '&key=' + API_key
    # Create a custom SSL context with verification disabled
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    jsonResult = urllib.request.urlopen(request, context=context).read()
    response = json.loads(jsonResult)
    return response



def build_distance_matrix(response):
    distance_matrix = []
    for row in response['rows']:
        row_list = [row['elements'][j]['distance']['value'] for j in range(len(row['elements']))]
        distance_matrix.append(row_list)
    return distance_matrix

########
# Main #
########
def return_distance_matrix():
    """Entry point of the program"""
    # Create the data.
    data = create_data()
    # addresses = data['addresses']
    # API_key = data['API_key']
    distance_matrix = create_distance_matrix(data)
    return distance_matrix




###
# Pi specific imports
##
import requests
import time
import collections
import csv
import os
# Constants

def load_csv(csv_path: str = "../FinalLEDTable_with_ids.csv", zero_indexed: bool = False):
    loaded = 0
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # Verify columns
        header = [c.lower() for c in reader.fieldnames or []]
        if 'index' not in header or 'station_id' not in header:
            raise ValueError("CSV must include 'index' and 'station_id' columns")
        for row in reader:
            idx = row['index']
            pos = int(idx) - 1 if zero_indexed else int(idx)
            station_id = row['station_id']
            DATA_DICT[station_id] = {
                'position': pos,
                'bikes_available': None,
                'docks_available': None,
                'ebikes_available': None
            }
            loaded += 1
    return loaded

def get_data():
    try:
        response = requests.get(STATION_STATUS_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Failed to fetch station data: {e}")
        return None
    stations = data["data"]["stations"]
    for station in stations:
        station_id = station["station_id"]
        new_bikes = station["num_bikes_available"]
        new_ebikes = station["num_ebikes_available"]
        new_docks = station["num_docks_available"]
        prev_station = DATA_DICT[station_id]
        if (prev_station["bikes_available"] > new_bikes):
            UPDATE_LIST.append((prev_station["position"], COLOR_MAP["red"]))
            DATA_DICT[station_id] = {
                "bikes_available": new_bikes,
                "docks_available": new_docks,
                "ebikes_available": new_ebikes
            }
        elif (prev_station["bikes_available"] < new_bikes):
            if (new_ebikes > prev_station["ebikes_available"]):
                UPDATE_LIST.append((prev_station["position"], COLOR_MAP["green"]))
            else:
                UPDATE_LIST.append((prev_station["position"], COLOR_MAP["blue"]))
            DATA_DICT[station_id] = {
                "bikes_available": new_bikes,
                "docks_available": new_docks,
                "ebikes_available": new_ebikes
            }

##Blinks them, and then leaves them on the last color
def update_leds():
    for _ in range(NUM_BLINKS):
        for (led_idx, j) in UPDATE_LIST:
                LEDS[led_idx] = (0,0,0)
        time.sleep(BLINK_DURATION/2)
        for (led_idx, j) in UPDATE_LIST:
            LEDS[led_idx] = j
        time.sleep(BLINK_DURATION/2)
    ##Final color: Should be a gradient given how many bikes there are of each type 
    for i, _ in UPDATE_LIST:
        pass
    UPDATE_LIST.clear()



STATION_STATUS_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"
COLOR_MAP = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255)
}
DATA_DICT = {}
UPDATE_LIST = []
LEDS = []
NUM_BLINKS=3
BLINK_DURATION=0.5
UPDATE_RATE = 1 # Seconds between update

if __name__ == '__main__':
    load_csv(DATA_DICT)
    while True:
        s_time = time.time()
        get_data()
        update_leds()
        time_dormant = max(0, UPDATE_RATE - (time.time() - s_time))
        time.sleep(time_dormant)


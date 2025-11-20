import requests
import os


def get_station_ids(of):
    url = "https://gbfs.lyft.com/gbfs/2.3/bkn/fr/station_information.json"
    r = requests.get(url)
    dict = r.json()
    stations = dict["data"]["stations"]
    with open(of, "w") as f:
        for station in stations:
            f.write(f"{station['station_id']}\n")


def load_station_ids(in_f):
    with open(in_f, "r") as f:
        station_ids = [line.strip() for line in f]
    return station_ids


if __name__ == "__main__":
    url = "https://gbfs.lyft.com/gbfs/2.3/bkn/fr/station_information.json"
    of = "station_data.txt"
    r = requests.get(url)
    dict = r.json()
    stations = dict["data"]["stations"]
    with open(of, "w") as f:
        for station in stations:
            f.write(
                f"{station['station_id']} \t {station['lat']} \t {station['lon']} \n"
            )

"""
API keys extracted from the original project.
If you prefer, replace the values here with environment variables or a .env file.
"""

# Database connection settings
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "biketrack_db"
DB_USER = "biketracker"
DB_PASSWORD = "biketrack_password"  # Change this to a secure password in production

# Metadata types
LIVE = 1
ROUTE = 2
HISTORIC = 3

# Update interval in seconds
UPDATE_INTERVAL = 30
TRIP_UPDATE_INTERVAL = 60 * 60 * 24 * 30  # Every 30 Days


# Constants
COLOR_MAP = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
    "blank": (0, 0, 0),
}


# The LED behavior depends on the driver itself...
# LEDS = [0] * 700  # placeholder for LED strip
NUM_BLINKS = 5
BLINK_DURATION = 1
UPDATE_RATE = 1  # Seconds between update

N_LEDS = 665

S3_BUCKET_URL = "https://s3.amazonaws.com/tripdata"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
INFO_URL = "https://gbfs.lyft.com/gbfs/2.3/bkn/en/station_information.json"

# 50 distinct colors for historic trip rendering
HISTORIC_STATION_THRESHOLD = 200  # meters
HISTORIC_TRIP_WINDOWS_MINUTES = [30, 60, 120]
HISTORIC_TRIP_CANDIDATE_MULTIPLIER = 5
HISTORIC_TRIP_MAX_QUERY_LIMIT = 250
HISTORIC_COLORS = [
    "#FF0000",
    "#00FF00",
    "#0000FF",
    "#FFFF00",
    "#FF00FF",
    "#00FFFF",
    "#FF8000",
    "#FF0080",
    "#80FF00",
    "#00FF80",
    "#0080FF",
    "#8000FF",
    "#FF4000",
    "#FF0040",
    "#40FF00",
    "#00FF40",
    "#0040FF",
    "#4000FF",
    "#FFC000",
    "#FF00C0",
    "#C0FF00",
    "#00FFC0",
    "#00C0FF",
    "#C000FF",
    "#FF6600",
    "#FF0066",
    "#66FF00",
    "#00FF66",
    "#0066FF",
    "#6600FF",
    "#FFAA00",
    "#FF00AA",
    "#AAFF00",
    "#00FFAA",
    "#00AAFF",
    "#AA00FF",
    "#FF3333",
    "#33FF33",
    "#3333FF",
    "#FFFF33",
    "#FF33FF",
    "#33FFFF",
    "#FF6633",
    "#33FF66",
    "#6633FF",
    "#FFCC33",
    "#CC33FF",
    "#33FFCC",
    "#FF9933",
    "#9933FF",
]

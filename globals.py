"""
API keys extracted from the original project.
If you prefer, replace the values here with environment variables or a .env file.
"""

# Database connection settings
DB_HOST = "100.64.0.3"
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
STATION_STATUS_URL = "https://gbfs.lyft.com/gbfs/2.3/bkn/en/station_status.json"

# Distinct colors for historic trip rendering (up to 20 concurrent trips)
HISTORIC_STATION_THRESHOLD = 200  # meters
HISTORIC_TRIP_WINDOWS_MINUTES = [30, 60, 120]
HISTORIC_TRIP_CANDIDATE_MULTIPLIER = 5
HISTORIC_TRIP_MAX_QUERY_LIMIT = 20
HISTORIC_COLORS = [
    "#E6194B",  # red
    "#3CB44B",  # green
    "#4363D8",  # blue
    "#F58231",  # orange
    "#911EB4",  # purple
    "#46F0F0",  # cyan
    "#F032E6",  # magenta
    "#BCF60C",  # lime
    "#FABED4",  # pink
    "#008080",  # teal
    "#E6BEFF",  # lavender
    "#9A6324",  # brown
    "#FFFAC8",  # beige
    "#800000",  # maroon
    "#AAFFC3",  # mint
    "#808000",  # olive
    "#FFD8B1",  # apricot
    "#000075",  # navy
    "#808080",  # gray
    "#FF4500",  # orange red
]

"""
API keys extracted from the original project.
If you prefer, replace the values here with environment variables or a .env file.
"""

# Database connection settings
DB_HOST = 'localhost'
DB_PORT = 5432
DB_NAME = 'biketrack_db'
DB_USER = 'biketracker'
DB_PASSWORD='biketrack_password'  # Change this to a secure password in production

# Metadata types
LIVE=1
ROUTE=2
HISTORIC=3

# Update interval in seconds
UPDATE_INTERVAL = 30

TRIP_UPDATE_INTERVAL = 60 * 60  * 24 * 30 # Every 30 Days

S3_BUCKET_URL = "https://s3.amazonaws.com/tripdata"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

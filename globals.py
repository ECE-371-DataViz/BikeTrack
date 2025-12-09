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
LIVE=0
ROUTE=1
HISTORIC=2

# Update interval in seconds
UPDATE_INTERVAL = 30

HISTORY_PERIOD = 5 # Minutes between historic data snapshots

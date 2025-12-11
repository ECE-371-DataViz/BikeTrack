#!/bin/bash

set +e  # Don't exit on errors - we need to handle the verification check

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Database credentials - match globals.py
DB_HOST='localhost'
DB_PORT=5432
DB_NAME='biketrack_db'
DB_USER='biketracker'
DB_PASSWORD='biketrack_password'  # Change this to a secure password in production

echo -e "${BLUE}=== PostgreSQL Setup for BikeTrack ===${NC}"

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo -e "${RED}PostgreSQL is not installed. Installing...${NC}"
    sudo apt update
    sudo apt install -y postgresql postgresql-contrib
fi

# Start PostgreSQL service
echo -e "${BLUE}Starting PostgreSQL service...${NC}"
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Wait for PostgreSQL to be ready
echo -e "${BLUE}Waiting for PostgreSQL to be ready...${NC}"
sleep 2

# Fix authentication method for localhost connections (ident -> md5)
echo -e "${BLUE}Configuring PostgreSQL authentication...${NC}"
HBA_CONF=$(sudo -u postgres psql -t -c "SELECT setting FROM pg_settings WHERE name = 'config_file';" | xargs)
if [ -f "$HBA_CONF" ]; then
    sudo sed -i "s/^host.*127.0.0.1.*ident/host    all             all             127.0.0.1\/32            md5/" "$HBA_CONF"
    sudo sed -i "s/^host.*::1.*ident/host    all             all             ::1\/128                 md5/" "$HBA_CONF"
    sudo systemctl reload postgresql
    sleep 1
fi

# Create database and user
echo -e "${BLUE}Creating database user and database...${NC}"

# Create user with password (using sudo to run as postgres user)
sudo -u postgres psql << 'EOFDB'
-- Drop existing user and database if they exist (for fresh setup)
DROP DATABASE IF EXISTS biketrack_db;
DROP USER IF EXISTS biketracker;

-- Create new user with password
CREATE USER biketracker WITH PASSWORD 'biketrack_password';

-- Create database owned by the user
CREATE DATABASE biketrack_db OWNER biketracker;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE biketrack_db TO biketracker;

-- Connect to the database and set schema privileges
\c biketrack_db
GRANT ALL PRIVILEGES ON SCHEMA public TO biketracker;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO biketracker;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO biketracker;
EOFDB

# Verify connection
echo -e "${BLUE}Verifying database connection...${NC}"
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT version();" 2>&1 | head -1

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ PostgreSQL database and user created successfully${NC}"
    echo -e "${GREEN}✓ Database: $DB_NAME${NC}"
    echo -e "${GREEN}✓ User: $DB_USER${NC}"
    echo -e "${GREEN}✓ Host: $DB_HOST:$DB_PORT${NC}"
    echo ""
    echo -e "${BLUE}Connection string:${NC}"
    echo "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"
    echo ""
    echo -e "${BLUE}Update globals.py with the following if needed:${NC}"
    echo "DB_HOST = '$DB_HOST'"
    echo "DB_PORT = $DB_PORT"
    echo "DB_NAME = '$DB_NAME'"
    echo "DB_USER = '$DB_USER'"
    echo "DB_PASSWORD = '$DB_PASSWORD'"
    echo ""
    echo -e "${GREEN}PostgreSQL setup completed!${NC}"
else
    echo -e "${RED}✗ Failed to verify database connection${NC}"
    echo -e "${RED}The database may have been created, but connection verification failed.${NC}"
    echo -e "${RED}Try running this command manually to debug:${NC}"
    echo "PGPASSWORD='$DB_PASSWORD' psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c \"SELECT version();\""
    exit 1
fi

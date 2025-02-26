#!/bin/bash

# Check if venv exists and remove it if it does
if [ -d "backend/venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf backend/venv
fi

# Create virtual environment
echo "Creating virtual environment for backend..."
cd backend
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "Please update the .env file with your API keys and credentials."
fi

echo "Backend setup complete!"
echo "To activate the environment, run: cd backend && source venv/bin/activate"
echo "To start the backend server, run: uvicorn app.main:app --reload" 
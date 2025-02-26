#!/bin/bash

# Create virtual environment
echo "Creating virtual environment for frontend..."
cd frontend
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Frontend setup complete!"
echo "To activate the environment, run: cd frontend && source venv/bin/activate"
echo "To start the frontend app, run: streamlit run app.py" 
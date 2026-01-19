A full-stack web application for tracking personal finances with intelligent transaction
categorization using machine learning. Built with FastAPI, vanilla JavaScript, and SQLite.


CSV Upload: Import transactions from any bank or credit card CSV file
ML-Powered Categorization: Automatically categorizes transactions into 8 categories using scikit-learn
Smart Budget Analysis: Based on the 50-30-20 budgeting rule

50% Needs (Groceries, Utilities, Transportation, Health)
30% Savings & Investments
20% Wants (Dining, Entertainment, Shopping)


Interactive Visualizations:

Spending by category (doughnut chart)
Monthly spending trends (line chart)


Self-Learning: Model improves accuracy as you upload more transactions


Backend

FastAPI - Modern Python web framework
SQLite - Lightweight database
scikit-learn - Machine learning for transaction categorization
pandas - CSV processing and data manipulation

Frontend

Vanilla JavaScript - No framework dependencies
Chart.js - Data visualizations
HTML/CSS - Responsive UI

Setup Instructions
Prerequisites

Python 3.8 or higher
pip (Python package manager)

Installation

Clone the repository

bashgit clone <your-repo-url>
cd personal-finance-dashboard

Create virtual environment

bashpython -m venv venv

# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate

Install dependencies

bashpip install -r requirements.txt

Run the backend server

bashpython main.py
The API will be available at http://localhost:8000

Open the frontend

Simply open index.html in your web browser, or serve it with:
bashpython -m http.server 3000
Then visit http://localhost:3000
Usage

Prepare your CSV file with columns for:

Date (any column with "date" in the name)
Description (any column with "desc" or "name")
Amount (any column with "amount" or "price")


Upload via the web interface

Click "Choose CSV File"
Select your transaction file
Click "Upload & Analyze"


View insights

Category breakdown chart
Monthly spending trends
Budget recommendations based on 50-30-20 rule
Recent transactions table




# Property Price Analysis & Prediction Tool

![Real Estate](https://img.shields.io/badge/ML-Real%20Estate-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.15+-red)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange)

## 📊 Overview

Interactive property price analytics and prediction tool for real estate decisions. This Streamlit application analyzes property data based on location and various characteristics, helping real estate professionals, investors, and home buyers make data-driven decisions through comprehensive analytics and price predictions.

## ✨ Features

- **Data Analysis**: Upload and analyze your property dataset with comprehensive visualization
- **Location-Based Analysis**: Filter and analyze properties by state, location, and more
- **Interactive Visualizations**: 
  - Price distribution plots
  - Price vs Area scatter plots
  - Year-wise price trend analysis
  - Enterprise comparison charts
- **Price Prediction**: Machine learning model to estimate property prices based on various features
- **User-Friendly Filters**: Easily segment data by bedrooms, furnishing status, price range, etc.
- **Detailed Property Listings**: View and sort complete property details
- **Year-wise Predictions**: Forecast property prices for future years

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required libraries: streamlit, pandas, numpy, plotly, scikit-learn

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/property-price-analysis.git
cd property-price-analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

5. Open your browser and go to `http://localhost:8501`

## 📝 Usage Guide

### Data Analysis & Visualization

1. Upload your property data CSV file through the file uploader
2. Use the sidebar filters to segment data by:
   - Location (State and specific locality)
   - Price range
   - Bedrooms configuration (1 RK, 1 BHK, 2 BHK, etc.)
   - Furnishing status
   - Enterprise/Builder
3. Explore the interactive charts and property listings including:
   - Market overview with key metrics
   - Year-wise price analysis
   - Price distribution
   - Price vs Area relationship

### Price Prediction

1. Click "Train Price Prediction Model" to build the machine learning model
2. Enter property details:
   - Area (sqft)
   - Bedrooms configuration
   - Bathrooms
   - Floor and total floors
   - Age and parking
   - Metro distance
   - Furnishing status and facing direction
   - Location
3. Click "Estimate Price" to get the predicted property price

## 📊 Required Data Format

Your CSV file should include these columns (not all are mandatory):

| Column | Description |
|--------|-------------|
| location | Property location (area/locality) |
| state | State/Province of the property |
| price | Property price (in lakhs) |
| sqft | Property area in square feet |
| bedrooms | Number of bedrooms (can be 1 RK, 1 BHK, etc.) |
| bathrooms | Number of bathrooms |
| floor | Floor number of the property |
| total_floors | Total number of floors in the building |
| age_years | Age of the property in years |
| parking | Number of parking spots |
| metro_distance_km | Distance to nearest metro station in km |
| furnishing | Furnishing status (Unfurnished, Semi-Furnished, Fully Furnished) |
| facing | Direction the property faces (North, South, etc.) |
| enterprise | Builder/Developer name |
| price_per_sqft | Price per square foot |

Example row:
```
Whitefield,Karnataka,85.5,1200,2,2,3,5,2,1,1.5,Unfurnished,North,BuilderA,7125
```

## 🔍 How It Works

1. **Data Processing**: 
   - The app loads and processes property data
   - Handles data type conversions (e.g., converting bedroom values to RK/BHK format)
   - Calculates additional metrics like price per square foot

2. **Location Analysis**: 
   - Calculates location-based statistics
   - Groups properties by location and state for comparison

3. **Machine Learning**: 
   - Uses Linear Regression to predict property prices
   - Features are standardized for better model performance
   - Handles both numeric and categorical features with one-hot encoding

4. **Visualization**: 
   - Creates interactive visualizations using Plotly
   - Offers year-wise trend analysis
   - Provides price distribution and correlation plots

## 🛠️ Machine Learning Model

The application uses a Linear Regression model to predict property prices based on:
- Numeric features: area, bedrooms, bathrooms, floor, etc.
- Categorical features: location, state, furnishing status, etc.
- All features are standardized using StandardScaler for better performance

## 🔮 Future Enhancements

- Adding more advanced ML models (Random Forest, XGBoost)
- Time series forecasting for future price trends
- User authentication and saved preferences
- Export functionality for reports and predictions
- Interactive map visualization for location-based analysis

## 📞 Contact

Your Name -https://www.linkedin.com/in/om-shikhare?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app

Project Link: https://property-price-analysis-zzbdexrtu3pclk3ku42zxx.streamlit.app/

---

# Property-Price-Analysis Tool 🏠
Interactive nature, Property price focus, Analytics and prediction capabilities, Real estate domain, Decision-making utility

A robust Streamlit web application for analyzing and predicting property prices based on location and various property characteristics. This tool helps real estate professionals, investors, and home buyers make data-driven decisions by providing comprehensive analytics and price predictions.

## Features

- **Location-Based Analysis**: Filter and analyze properties by state, location, and more
- **Price Prediction**: Machine learning model to estimate property prices based on various features
- **Interactive Visualizations**: Dynamic charts and graphs for market insights
- **Year-wise Trend Analysis**: Track property price trends over time
- **Enterprise Comparison**: Compare property prices across different enterprises/builders
- **User-Friendly Filters**: Easily segment data by bedrooms, furnishing status, price range, etc.
- **Detailed Property Listings**: View and sort complete property details

## 📊 Screenshots

*[Insert screenshots of your application here]*

## 🛠️ Tech Stack

- **Streamlit**: For the interactive web interface
- **Pandas**: For data manipulation and analysis
- **NumPy**: For numerical computations
- **Plotly Express**: For interactive data visualizations
- **Scikit-learn**: For machine learning models (Linear Regression)
- **StandardScaler**: For feature scaling

## 🔧 Installation & Setup

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
   streamlit run main.py
   ```

## 📁 Required Data Format

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

## 🚀 Usage

1. Upload your property data CSV file through the file uploader
2. Use the sidebar filters to segment data by location, price range, bedrooms, etc.
3. Explore the interactive charts and property listings
4. Train the price prediction model and get estimates for specific property configurations

## 🧠 Machine Learning Model

The application uses a Linear Regression model to predict property prices based on:
- Numeric features (area, bedrooms, bathrooms, floor, etc.)
- Categorical features (location, state, furnishing status, etc.) using one-hot encoding
- Standardized scaling for better model performance

## 🔮 Future Enhancements

- Adding more advanced ML models (Random Forest, XGBoost)
- Time series forecasting for future price trends
- User authentication and saved preferences
- Export functionality for reports and predictions
- Interactive map visualization for location-based analysis

## 📞 Contact

Your Name -([https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/om-shikhare)](https://www.linkedin.com/in/om-shikhare?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)

Project Link:()

---

⭐ Star this repo if you find it useful!

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class LocationBasedHouseAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.location_price_data = {}
        self.year_price_model = None
        self.features = []
        
    def load_and_process_data(self, uploaded_file):
        df = pd.read_csv(uploaded_file)
        
        numeric_columns = ['price', 'sqft', 'bedrooms', 'bathrooms', 'floor', 
                         'total_floors', 'age_years', 'parking', 'metro_distance_km', 
                         'price_per_sqft']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert bedroom values to RK/BHK format
        if 'bedrooms' in df.columns:
            bedroom_mapping = {
                0: '1 RK',
                1: '1 BHK',
                2: '2 BHK',
                3: '3 BHK',
                4: '4 BHK',
                5: '5 BHK'
            }
            df['bedrooms'] = df['bedrooms'].map(lambda x: bedroom_mapping.get(x, f"{int(x)} BHK" if pd.notnull(x) and x > 5 else x))

        # Calculate year from age_years
        if 'age_years' in df.columns:
            current_year = pd.Timestamp.now().year
            df['year'] = current_year - df['age_years']
        
        df = df.dropna(subset=['price'])
        self.calculate_location_stats(df)
        self.train_year_price_model(df)
        return df
    
    def train_year_price_model(self, df):
        if 'year' in df.columns and len(df) > 0:
            X = df[['year']].values.reshape(-1, 1)
            y = df['price'].values
            self.year_price_model = LinearRegression()
            self.year_price_model.fit(X, y)
    
    def predict_price_for_year(self, year):
        if self.year_price_model is None:
            raise ValueError("Year price model not trained yet")
        return self.year_price_model.predict([[year]])[0]
    
    def calculate_location_stats(self, df):
        if 'location' in df.columns:
            self.location_price_data['location'] = df.groupby('location').agg({
                'price': ['mean', 'median', 'min', 'max', 'count'],
                'price_per_sqft': ['mean', 'median']
            }).round(2)
        
        if 'state' in df.columns:    
            self.location_price_data['state'] = df.groupby('state').agg({
                'price': ['mean', 'median', 'min', 'max', 'count'],
                'price_per_sqft': ['mean', 'median']
            }).round(2)
    
    def train_location_model(self, df):
        df_model = df.copy()
        
        # Convert BHK/RK back to numeric for model
        if 'bedrooms' in df_model.columns and isinstance(df_model['bedrooms'].iloc[0], str):
            bedroom_mapping = {
                '1 RK': 0,
                '1 BHK': 1,
                '2 BHK': 2,
                '3 BHK': 3,
                '4 BHK': 4,
                '5 BHK': 5
            }
            df_model['bedrooms'] = df_model['bedrooms'].map(lambda x: bedroom_mapping.get(x, int(x.split()[0]) if isinstance(x, str) and x.split()[0].isdigit() else np.nan))
        
        numeric_features = [col for col in ['sqft', 'bedrooms', 'bathrooms', 'floor', 'total_floors', 
                          'age_years', 'parking', 'metro_distance_km'] if col in df_model.columns]
        
        categorical_features = [col for col in ['state', 'location', 'furnishing', 'facing'] 
                               if col in df_model.columns]
        
        if not numeric_features or len(df_model) == 0:
            raise ValueError("Not enough numeric features or data to train the model")
        
        # Prepare features and target
        X = df_model[numeric_features].copy()
        X_with_cat = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)
        
        # Keep only numeric columns for scaling
        X_numeric_cols = X.columns.tolist()
        
        # Get all dummy columns
        dummy_cols = [col for col in X_with_cat.columns if col not in df_model.columns]
        
        # Combine numeric and dummy columns
        X = X_with_cat[X_numeric_cols + dummy_cols]
        y = df_model['price']
        
        # Store feature names for prediction
        self.features = X.columns.tolist()
        
        # Scale features
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Split data and train model
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        return {
            'score': self.model.score(X_test, y_test),
            'features': X.columns.tolist()
        }
    
    def predict_price(self, input_data):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Convert BHK/RK to numeric if needed
        if 'bedrooms' in input_data.columns and isinstance(input_data['bedrooms'].iloc[0], str):
            bedroom_mapping = {
                '1 RK': 0,
                '1 BHK': 1,
                '2 BHK': 2,
                '3 BHK': 3,
                '4 BHK': 4,
                '5 BHK': 5
            }
            input_data['bedrooms'] = input_data['bedrooms'].map(lambda x: bedroom_mapping.get(x, int(x.split()[0]) if isinstance(x, str) and x.split()[0].isdigit() else np.nan))
            
        # Create dummy variables
        categorical_features = [col for col in ['state', 'location', 'furnishing', 'facing'] 
                               if col in input_data.columns]
        input_df = pd.get_dummies(input_data, columns=categorical_features, drop_first=True)
        
        # Align input columns with model features
        missing_cols = set(self.features) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
            
        # Ensure correct column order
        input_df = input_df[self.features]
        input_scaled = self.scaler.transform(input_df)
        
        return self.model.predict(input_scaled)[0]

def run_location_analysis():
    st.set_page_config(page_title="Property Price Analysis", layout="wide")
    
    analyzer = LocationBasedHouseAnalyzer()
    
    st.title("🏠 Property Price Analysis")
    st.markdown("""
    Analyze property prices based on location and property characteristics.
    """)
    
    uploaded_file = st.file_uploader("Upload your property dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = analyzer.load_and_process_data(uploaded_file)
            
            st.sidebar.header("📍 Location Filters")
            
            state_column_exists = 'state' in df.columns
            location_column_exists = 'location' in df.columns
            
            if state_column_exists:
                selected_state = st.sidebar.selectbox(
                    "Select State",
                    options=['All'] + sorted(df['state'].unique().tolist())
                )
                
                if selected_state != 'All':
                    df_filtered = df[df['state'] == selected_state]
                    
                    if location_column_exists:
                        selected_location = st.sidebar.selectbox(
                            "Select Location",
                            options=['All'] + sorted(df_filtered['location'].unique().tolist())
                        )
                        
                        if selected_location != 'All':
                            df_filtered = df_filtered[df_filtered['location'] == selected_location]
                else:
                    df_filtered = df
            else:
                df_filtered = df
                if location_column_exists:
                    selected_location = st.sidebar.selectbox(
                        "Select Location",
                        options=['All'] + sorted(df['location'].unique().tolist())
                    )
                    
                    if selected_location != 'All':
                        df_filtered = df_filtered[df_filtered['location'] == selected_location]
            
            st.sidebar.header("🏷️ Property Filters")
            
            # Add Enterprise and Price Range filters
            min_price = float(df['price'].min())
            max_price = float(df['price'].max())
            
            # Modified Enterprise selection to include price
            if 'enterprise' in df.columns:
                enterprise_price_data = df.groupby('enterprise')['price'].mean().round(2)
                enterprise_options = ['All'] + [f"{enterprise} (₹{price:.2f}L)" 
                                             for enterprise, price in enterprise_price_data.items()]
                
                selected_enterprise_with_price = st.sidebar.selectbox(
                    "Select Enterprise",
                    options=enterprise_options
                )
                
                # Extract enterprise name without price for filtering
                selected_enterprise = 'All' if selected_enterprise_with_price == 'All' else \
                                    selected_enterprise_with_price.split(' (')[0]
            else:
                selected_enterprise = 'All'
            
            price_range = st.sidebar.slider(
                "Price Range (in Lakhs)",
                min_value=min_price,
                max_value=max_price,
                value=(min_price, max_price)
            )
            
            if 'bedrooms' in df.columns:
                bedrooms_options = ['All']
                if '1 RK' in df['bedrooms'].values:
                    bedrooms_options.append('1 RK')
                if '1 BHK' in df['bedrooms'].values:
                    bedrooms_options.append('1 BHK')
                    
                bhk_options = sorted([opt for opt in df['bedrooms'].unique() 
                                   if isinstance(opt, str) and 'BHK' in opt and opt != '1 BHK'],
                                   key=lambda x: int(x.split()[0]) if x.split()[0].isdigit() else 0)
                bedrooms_options.extend(bhk_options)
                
                selected_bedrooms = st.sidebar.selectbox(
                    "Select Bedrooms",
                    options=bedrooms_options
                )
            else:
                selected_bedrooms = 'All'
            
            if 'furnishing' in df.columns:
                selected_furnishing = st.sidebar.multiselect(
                    "Furnishing Status",
                    options=sorted(df['furnishing'].unique().tolist())
                )
            else:
                selected_furnishing = []
            
            if selected_enterprise != 'All' and 'enterprise' in df.columns:
                df_filtered = df_filtered[df_filtered['enterprise'] == selected_enterprise]
                
            # Apply price range filter
            df_filtered = df_filtered[
                (df_filtered['price'] >= price_range[0]) & 
                (df_filtered['price'] <= price_range[1])
            ]
                
            if selected_bedrooms != 'All' and 'bedrooms' in df.columns:
                df_filtered = df_filtered[df_filtered['bedrooms'] == selected_bedrooms]
            
            if selected_furnishing and 'furnishing' in df.columns:
                df_filtered = df_filtered[df_filtered['furnishing'].isin(selected_furnishing)]
            
            st.header("📊 Market Overview")
            
            # Add Year Price Prediction at the top
            if 'year' in df.columns and analyzer.year_price_model is not None:
                col_year = st.columns(1)[0]
                with col_year:
                    prediction_year = st.number_input("Enter Year for Price Prediction", 
                                                   min_value=2000, 
                                                   max_value=2050, 
                                                   value=2025)
                    if st.button("Predict Price for Year"):
                        try:
                            predicted_price = analyzer.predict_price_for_year(prediction_year)
                            st.metric("Predicted Average Price", f"₹{predicted_price:.2f}L")
                        except Exception as e:
                            st.error(f"Prediction error: {str(e)}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Price", f"₹{df_filtered['price'].mean():,.2f}L")
            with col2:
                if 'price_per_sqft' in df_filtered.columns:
                    st.metric("Avg Price/sqft", f"₹{df_filtered['price_per_sqft'].mean():,.0f}")
                else:
                    st.metric("Avg Price/sqft", "N/A")
            with col3:
                if 'sqft' in df_filtered.columns:
                    st.metric("Avg Area", f"{df_filtered['sqft'].mean():,.0f} sqft")
                else:
                    st.metric("Avg Area", "N/A")
            with col4:
                st.metric("Properties", f"{len(df_filtered):,}")
            
            # Year-wise price analysis
            if 'year' in df_filtered.columns:
                st.subheader("Year-wise Price Analysis")
                yearly_avg_price = df_filtered.groupby('year')['price'].mean().reset_index()
                fig_yearly = px.bar(yearly_avg_price, 
                                  x='year', 
                                  y='price',
                                  title="Average Property Price by Year",
                                  labels={'year': 'Year', 'price': 'Average Price (Lakhs)'})
                st.plotly_chart(fig_yearly, use_container_width=True)
            
            st.subheader("Price Distribution")
            fig_price = px.histogram(df_filtered, x="price", 
                                   title="Price Distribution",
                                   labels={"price": "Price (Lakhs)", "count": "Number of Properties"})
            st.plotly_chart(fig_price, use_container_width=True)
            
            if 'sqft' in df_filtered.columns:
                st.subheader("Price vs Area")
                scatter_color = "bedrooms" if "bedrooms" in df_filtered.columns else None
                scatter_size = "price_per_sqft" if "price_per_sqft" in df_filtered.columns else None
                
                hover_data = []
                if "location" in df_filtered.columns:
                    hover_data.append("location")
                if "furnishing" in df_filtered.columns:
                    hover_data.append("furnishing")
                
                fig_scatter = px.scatter(df_filtered, x="sqft", y="price",
                                       color=scatter_color, size=scatter_size,
                                       hover_data=hover_data,
                                       title="Price vs Area by Bedrooms")
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Enterprise-specific price list
            if 'enterprise' in df.columns and selected_enterprise != 'All':
                st.subheader(f"Price List for {selected_enterprise}")
                display_cols = ['price', 'sqft']
                if 'location' in df_filtered.columns:
                    display_cols.insert(0, 'location')
                if 'bedrooms' in df_filtered.columns:
                    display_cols.append('bedrooms')
                display_cols.append('enterprise')
                if 'price_per_sqft' in df_filtered.columns:
                    display_cols.append('price_per_sqft')
                
                enterprise_df = df_filtered[display_cols].sort_values('price', ascending=False)
                
                format_dict = {'price': '₹{:,.2f}L'}
                if 'sqft' in enterprise_df.columns:
                    format_dict['sqft'] = '{:,.0f}'
                if 'price_per_sqft' in enterprise_df.columns:
                    format_dict['price_per_sqft'] = '₹{:,.0f}'
                
                st.dataframe(enterprise_df.style.format(format_dict))
            
            st.header("🏠 Property Listings")
            
            # Determine which columns to display
            all_possible_columns = ['location', 'price', 'sqft', 'bedrooms', 'bathrooms', 
                                  'furnishing', 'floor', 'total_floors', 'parking', 
                                  'facing', 'metro_distance_km', 'price_per_sqft']
            display_columns = [col for col in all_possible_columns if col in df_filtered.columns]
            
            if 'enterprise' in df.columns:
                if 'location' in display_columns:
                    idx = display_columns.index('location')
                    display_columns.insert(idx+1, 'enterprise')
                else:
                    display_columns.insert(0, 'enterprise')
            
            format_dict = {'price': '₹{:,.2f}L'}
            if 'sqft' in display_columns:
                format_dict['sqft'] = '{:,.0f}'
            if 'price_per_sqft' in display_columns:
                format_dict['price_per_sqft'] = '₹{:,.0f}'
            if 'metro_distance_km' in display_columns:
                format_dict['metro_distance_km'] = '{:.1f}'
            
            st.dataframe(
                df_filtered[display_columns]
                .sort_values('price', ascending=False)
                .style.format(format_dict)
            )
            
            st.header("🎯 Price Prediction")
            if st.button("Train Price Prediction Model"):
                with st.spinner("Training model..."):
                    try:
                        model_results = analyzer.train_location_model(df)
                        
                        st.subheader("Model Performance")
                        st.metric("R² Score", f"{model_results['score']:.3f}")
                        
                        st.subheader("Get Price Estimate")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            pred_sqft = st.number_input("Area (sqft)", min_value=100, value=1000)
                            
                            bedroom_options = ['1 RK', '1 BHK']
                            bedroom_options.extend([f"{i} BHK" for i in range(2, 6)])
                            
                            pred_bedrooms = st.selectbox("Bedrooms", 
                                                      options=bedroom_options,
                                                      index=1)
                            pred_bathrooms = st.number_input("Bathrooms", min_value=1, value=2)
                        
                        with col2:
                            pred_floor = st.number_input("Floor", min_value=1, value=5)
                            pred_total_floors = st.number_input("Total Floors", min_value=pred_floor, value=10)
                            pred_parking = st.number_input("Parking Spots", min_value=0, value=1)
                        
                        with col3:
                            pred_age = st.number_input("Age (years)", min_value=0, value=5)
                            pred_metro = st.number_input("Metro Distance (km)", min_value=0.0, value=2.0)
                            
                            # Only show if furnishing column exists
                            if 'furnishing' in df.columns:
                                pred_furnishing = st.selectbox("Furnishing", options=df['furnishing'].unique())
                            else:
                                pred_furnishing = "Unfurnished"  # Default value
                        
                        # Only show if facing column exists
                        if 'facing' in df.columns:
                            pred_facing = st.selectbox("Facing", options=df['facing'].unique())
                        else:
                            pred_facing = "North"  # Default value
                            
                        # Only show if location column exists
                        if 'location' in df.columns:
                            pred_location = st.selectbox("Location", options=df['location'].unique())
                        else:
                            pred_location = "Unknown Location"  # Default value
                            
                        if st.button("Estimate Price"):
                            input_data = {
                                'sqft': pred_sqft,
                                'bedrooms': pred_bedrooms,
                                'bathrooms': pred_bathrooms,
                                'floor': pred_floor,
                                'total_floors': pred_total_floors,
                                'age_years': pred_age,
                                'parking': pred_parking,
                                'metro_distance_km': pred_metro
                            }
                            
                            # Add categorical variables if they exist in the dataset
                            if 'location' in df.columns:
                                input_data['location'] = pred_location
                            
                            if 'state' in df.columns and 'location' in df.columns:
                                # Find the state for the selected location
                                location_states = df[df['location'] == pred_location]['state'].unique()
                                if len(location_states) > 0:
                                    input_data['state'] = location_states[0]
                                else:
                                    input_data['state'] = df['state'].iloc[0]  # Default to first state
                            
                            if 'furnishing' in df.columns:
                                input_data['furnishing'] = pred_furnishing
                                
                            if 'facing' in df.columns:
                                input_data['facing'] = pred_facing
                            
                            input_df = pd.DataFrame([input_data])
                            try:
                                predicted_price = analyzer.predict_price(input_df)
                                st.success(f"Estimated Price: ₹{predicted_price:.2f}L")
                            except Exception as e:
                                st.error(f"Prediction error: {str(e)}")
                                
                    except Exception as e:
                        st.error(f"Model training error: {str(e)}")
                        st.info("Make sure your dataset contains sufficient numeric features and samples.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please ensure your CSV file contains all required columns.")
    
    else:
        st.info("""
        Please upload a CSV file to begin the analysis.

        This is the data structure you need to upload:
        
        location,state,price,sqft,bedrooms,bathrooms,floor,total_floors,age_years,parking,metro_distance_km,furnishing,facing,enterprise,price_per_sqft
        
        Example row:
        Whitefield,Karnataka,85.5,1200,2,2,3,5,2,1,1.5,Unfurnished,North,BuilderA,7125
        """)

if __name__ == "__main__":
    run_location_analysis()
"""
Feature Engineering Script for Yango Mobility Prediction

This script contains all feature engineering functions to transform raw data
into ML-ready features for the Yango ride time prediction model.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import math
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature Engineering class for Yango mobility prediction
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.location_clusters = None
        self.fitted = False
        
    def create_datetime_features(self, df, datetime_col='lcl_start_transporting_dttm'):
        """
        Create comprehensive datetime features from trip start time
        """
        df = df.copy()
        
        # Ensure datetime column is in datetime format
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # Basic time features
        df['hour'] = df[datetime_col].dt.hour
        df['day_of_week'] = df[datetime_col].dt.dayofweek
        df['day_of_month'] = df[datetime_col].dt.day
        df['month'] = df[datetime_col].dt.month
        df['year'] = df[datetime_col].dt.year
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Rush hour indicators (morning and evening)
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
        
        # Peak hours (wider definition)
        df['is_peak_hour'] = ((df['hour'] >= 6) & (df['hour'] <= 10) | 
                             (df['hour'] >= 16) & (df['hour'] <= 20)).astype(int)
        
        # Night hours
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        
        # Business hours
        df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 17)).astype(int)
        
        # Lunch hours
        df['is_lunch_hours'] = ((df['hour'] >= 12) & (df['hour'] <= 14)).astype(int)
        
        # Cyclical features for time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_distance_features(self, df):
        """
        Create distance-related features including haversine distance
        """
        df = df.copy()
        
        # Haversine distance calculation
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate great circle distance between two points"""
            R = 6371  # Earth's radius in km
            
            # Convert to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            
            return R * c
        
        # Calculate haversine distance
        df['haversine_distance_km'] = df.apply(
            lambda row: haversine_distance(
                row['origin_lat'], row['origin_lon'],
                row['destination_lat'], row['destination_lon']
            ), axis=1
        )
        
        # Distance features if available
        if 'transporting_distance_fact_km' in df.columns:
            # Ratio of actual to straight-line distance (detour factor)
            df['detour_factor'] = df['transporting_distance_fact_km'] / (df['haversine_distance_km'] + 1e-6)
            
            # Distance difference
            df['distance_diff'] = df['transporting_distance_fact_km'] - df['haversine_distance_km']
            
            # Speed estimation (if we have the target)
            if 'transporting_time_fact_mnt' in df.columns:
                df['avg_speed_kmh'] = df['transporting_distance_fact_km'] / (df['transporting_time_fact_mnt'] / 60 + 1e-6)
        
        if 'str_distance_km' in df.columns:
            # Compare straight-line distance with haversine
            df['str_vs_haversine_ratio'] = df['str_distance_km'] / (df['haversine_distance_km'] + 1e-6)
        
        # Distance categories
        df['distance_category'] = pd.cut(
            df['haversine_distance_km'], 
            bins=[0, 1, 3, 5, 10, float('inf')],
            labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
        )
        
        return df
    
    def create_location_features(self, df, n_clusters=20):
        """
        Create location-based features using clustering
        """
        df = df.copy()
        
        # Combine origin and destination coordinates for clustering
        coords = np.vstack([
            df[['origin_lat', 'origin_lon']].values,
            df[['destination_lat', 'destination_lon']].values
        ])
        
        # Perform clustering if not already fitted
        if self.location_clusters is None:
            self.location_clusters = KMeans(n_clusters=n_clusters, random_state=42)
            self.location_clusters.fit(coords)
        
        # Assign cluster labels
        origin_clusters = self.location_clusters.predict(df[['origin_lat', 'origin_lon']])
        dest_clusters = self.location_clusters.predict(df[['destination_lat', 'destination_lon']])
        
        df['origin_cluster'] = origin_clusters
        df['dest_cluster'] = dest_clusters
        
        # Same cluster indicator
        df['same_cluster'] = (df['origin_cluster'] == df['dest_cluster']).astype(int)
        
        # Distance to city center (rough coordinates for Accra)
        accra_center_lat, accra_center_lon = 5.6037, -0.1870
        
        def distance_to_center(lat, lon):
            R = 6371  # Earth radius in km
            lat1, lon1, lat2, lon2 = map(math.radians, [lat, lon, accra_center_lat, accra_center_lon])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            return R * c
        
        df['origin_to_center_km'] = df.apply(
            lambda row: distance_to_center(row['origin_lat'], row['origin_lon']), axis=1
        )
        df['dest_to_center_km'] = df.apply(
            lambda row: distance_to_center(row['destination_lat'], row['destination_lon']), axis=1
        )
        
        # Movement direction (towards/away from center)
        df['movement_to_center'] = (df['dest_to_center_km'] < df['origin_to_center_km']).astype(int)
        
        # Bounding box features
        df['lat_range'] = np.abs(df['destination_lat'] - df['origin_lat'])
        df['lon_range'] = np.abs(df['destination_lon'] - df['origin_lon'])
        df['coord_range'] = df['lat_range'] + df['lon_range']
        
        return df
    
    def create_weather_features(self, df):
        """
        Create weather-related features
        """
        df = df.copy()
        
        # Handle missing weather data
        df['has_weather_data'] = df['temperature_C'].notna().astype(int)
        
        # Fill missing weather data with median values
        df['temperature_C'] = df['temperature_C'].fillna(df['temperature_C'].median())
        df['prev_hour_precipitation_mm'] = df['prev_hour_precipitation_mm'].fillna(0)
        
        # Rain indicators
        df['is_raining'] = (df['prev_hour_precipitation_mm'] > 0).astype(int)
        df['is_light_rain'] = ((df['prev_hour_precipitation_mm'] > 0) & 
                              (df['prev_hour_precipitation_mm'] <= 2.5)).astype(int)
        df['is_moderate_rain'] = ((df['prev_hour_precipitation_mm'] > 2.5) & 
                                 (df['prev_hour_precipitation_mm'] <= 7.5)).astype(int)
        df['is_heavy_rain'] = (df['prev_hour_precipitation_mm'] > 7.5).astype(int)
        
        # Temperature categories
        temp_median = df['temperature_C'].median()
        df['is_hot'] = (df['temperature_C'] > temp_median).astype(int)
        df['is_very_hot'] = (df['temperature_C'] > df['temperature_C'].quantile(0.75)).astype(int)
        df['is_cool'] = (df['temperature_C'] < df['temperature_C'].quantile(0.25)).astype(int)
        
        # Weather severity score
        df['weather_severity'] = (
            df['is_raining'] * 1 +
            df['is_moderate_rain'] * 2 +
            df['is_heavy_rain'] * 3 +
            df['is_very_hot'] * 1
        )
        
        # Encode precipitation type
        if 'precipitation_type' in df.columns:
            df['precipitation_type'] = df['precipitation_type'].fillna('No Rain')
            
            # Create binary features for precipitation types
            precip_types = df['precipitation_type'].unique()
            for ptype in precip_types:
                df[f'precip_{ptype.lower().replace(" ", "_")}'] = (df['precipitation_type'] == ptype).astype(int)
        
        return df
    
    def create_interaction_features(self, df):
        """
        Create interaction features between different feature groups
        """
        df = df.copy()
        
        # Time-Weather interactions
        df['rush_hour_rain'] = df['is_rush_hour'] * df['is_raining']
        df['weekend_rain'] = df['is_weekend'] * df['is_raining']
        df['hot_rush_hour'] = df['is_hot'] * df['is_rush_hour']
        
        # Distance-Time interactions
        df['long_distance_rush'] = (df['haversine_distance_km'] > 5) * df['is_rush_hour']
        df['short_distance_night'] = (df['haversine_distance_km'] <= 1) * df['is_night']
        
        # Weather-Distance interactions
        df['rain_long_distance'] = df['is_raining'] * (df['haversine_distance_km'] > 5)
        
        # Location-Time interactions
        df['center_rush_hour'] = (df['origin_to_center_km'] < 3) * df['is_rush_hour']
        df['same_area_weekend'] = df['same_cluster'] * df['is_weekend']
        
        return df
    
    def encode_categorical_features(self, df, categorical_cols=None):
        """
        Encode categorical features using label encoding
        """
        df = df.copy()
        
        if categorical_cols is None:
            categorical_cols = ['distance_category', 'origin_cluster', 'dest_cluster']
        
        # Only encode columns that exist in the dataframe
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                df[col] = df[col].astype(str)
                known_categories = set(self.label_encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if x in known_categories else 'unknown')
                
                # Add 'unknown' to label encoder if not present
                if 'unknown' not in self.label_encoders[col].classes_:
                    self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'unknown')
                
                df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def get_feature_columns(self):
        """
        Return list of all feature columns for modeling
        """
        base_features = [
            # Time features
            'hour', 'day_of_week', 'day_of_month', 'month',
            'is_weekend', 'is_rush_hour', 'is_morning_rush', 'is_evening_rush',
            'is_peak_hour', 'is_night', 'is_business_hours', 'is_lunch_hours',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            
            # Distance features
            'haversine_distance_km', 'distance_category',
            'origin_to_center_km', 'dest_to_center_km', 'movement_to_center',
            'lat_range', 'lon_range', 'coord_range',
            
            # Location features
            'origin_cluster', 'dest_cluster', 'same_cluster',
            
            # Weather features
            'temperature_C', 'prev_hour_precipitation_mm', 'has_weather_data',
            'is_raining', 'is_light_rain', 'is_moderate_rain', 'is_heavy_rain',
            'is_hot', 'is_very_hot', 'is_cool', 'weather_severity',
            
            # Interaction features
            'rush_hour_rain', 'weekend_rain', 'hot_rush_hour',
            'long_distance_rush', 'short_distance_night', 'rain_long_distance',
            'center_rush_hour', 'same_area_weekend'
        ]
        
        # Add conditional features
        conditional_features = [
            'detour_factor', 'distance_diff', 'avg_speed_kmh',
            'str_vs_haversine_ratio'
        ]
        
        return base_features, conditional_features
    
    def fit_transform(self, df):
        """
        Fit the feature engineer and transform the data
        """
        print("Starting feature engineering...")
        
        df = self.create_datetime_features(df)
        print("DateTime features created")
        
        df = self.create_distance_features(df)
        print("Distance features created")
        
        df = self.create_location_features(df)
        print("Location features created")
        
        df = self.create_weather_features(df)
        print("Weather features created")
        
        df = self.create_interaction_features(df)
        print("Interaction features created")
        
        df = self.encode_categorical_features(df)
        print("Categorical features encoded")
        
        self.fitted = True
        print("Feature engineering completed!")
        
        return df
    
    def transform(self, df):
        """
        Transform new data using fitted parameters
        """
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transforming new data")
        
        print("Transforming new data...")
        
        df = self.create_datetime_features(df)
        df = self.create_distance_features(df)
        df = self.create_location_features(df)
        df = self.create_weather_features(df)
        df = self.create_interaction_features(df)
        df = self.encode_categorical_features(df)
        
        print("Data transformation completed!")
        
        return df

def merge_with_weather(trip_df, weather_df):
    """
    Merge trip data with weather data
    """
    print("Merging trip data with weather data...")
    
    trip_df['lcl_start_transporting_dttm'] = pd.to_datetime(trip_df['lcl_start_transporting_dttm'])
    weather_df['lcl_datetime'] = pd.to_datetime(weather_df['lcl_datetime'])
    
    trip_df['trip_hour'] = trip_df['lcl_start_transporting_dttm'].dt.floor('H')
    weather_df['weather_hour'] = weather_df['lcl_datetime'].dt.floor('H')
    
    merged_df = trip_df.merge(
        weather_df[['weather_hour', 'precipitation_type', 'prev_hour_precipitation_mm', 'temperature_C']],
        left_on='trip_hour',
        right_on='weather_hour',
        how='left'
    )
    
    merged_df = merged_df.drop(['trip_hour', 'weather_hour'], axis=1)
    
    print(f"Merged {len(merged_df)} trips with weather data")
    print(f"Weather data coverage: {merged_df['temperature_C'].notna().mean()*100:.1f}%")
    
    return merged_df

def prepare_model_data(df, target_col='transporting_time_fact_mnt', is_training=True):
    """
    Prepare final dataset for modeling
    """
    print("Preparing data for modeling...")
    
    fe = FeatureEngineer()
    
    if is_training:
        df_processed = fe.fit_transform(df)
    else:
        df_processed = fe.transform(df)
    
    base_features, conditional_features = fe.get_feature_columns()
    
    available_features = [col for col in base_features if col in df_processed.columns]
    available_conditional = [col for col in conditional_features if col in df_processed.columns]
    
    all_features = available_features + available_conditional
    
    print(f"Selected {len(all_features)} features for modeling")
    
    if is_training:
        X = df_processed[all_features]
        y = df_processed[target_col] if target_col in df_processed.columns else None
        
        X = X.fillna(X.median())
        
        print(f"Training data shape: {X.shape}")
        if y is not None:
            print(f"Target data shape: {y.shape}")
        
        return X, y, fe, all_features
    else:
        X = df_processed[all_features]
        X = X.fillna(X.median())
        
        print(f"Test data shape: {X.shape}")
        
        return X, fe, all_features

if __name__ == "__main__":
    print("Feature Engineering Script for Yango Mobility Prediction")
    print("=" * 60)
    print("This script contains feature engineering functions.")
    print("Import this module in your notebooks to use the FeatureEngineer class.")
    print("=" * 60)

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Step 1: Data Collection and Analysis
data = pd.read_csv('CO2 Emissions_Canada.csv')
print(data)
# Statistical summary of numerical features
data.describe()


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of CO2 Emissions
plt.hist(data['CO2 Emissions(g/km)'], bins=20)
plt.xlabel('CO2 Emissions (g/km)')
plt.ylabel('Frequency')
plt.title('Distribution of CO2 Emissions')
plt.show()

# Scatter plot of Engine Size vs. Fuel Consumption
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Engine Size(L)', y='Fuel Consumption Comb (L/100 km)', data=data)
plt.xlabel('Engine Size (L)')
plt.ylabel('Fuel Consumption Comb (L/100 km)')
plt.title('Engine Size vs. Fuel Consumption')
plt.show()


# In[3]:


# Correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('CO2 Emissions_Canada.csv')

# Exclude non-numeric columns
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Step 1: Handling Missing Values
imputer = SimpleImputer(strategy='mean')
numeric_data_filled = imputer.fit_transform(numeric_data)
data_filled = pd.DataFrame(numeric_data_filled, columns=numeric_data.columns)

# Step 2: Handling Categorical Variables (if any)
# If there are categorical variables, you should encode them here

# Step 3: Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data_filled)

# Convert scaled_features to a DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=data_filled.columns)

# Step 4: Splitting Data
X = scaled_df.drop(columns=['CO2 Emissions(g/km)'])
y = scaled_df['CO2 Emissions(g/km)']
print(scaled_df)


# In[5]:


data = pd.read_csv('CO2 Emissions_Canada.csv')
X = data.drop(columns=['CO2 Emissions(g/km)'])
y = data['CO2 Emissions(g/km)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Drop non-numeric columns and categorical columns
X_train_numeric = X_train.select_dtypes(include=['float64', 'int64'])

# Handle categorical variables (one-hot encoding)
X_train_encoded = pd.get_dummies(X_train)

# Now, X_train_encoded contains only numerical features after one-hot encoding

# Step 3: Train ML Model for Prediction
# Using RandomForestRegressor for simplicity
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_encoded, y_train)
print(model)


# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import pickle

def preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Exclude non-numeric columns
    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    # Handling Missing Values
    imputer = SimpleImputer(strategy='mean')
    numeric_data_filled = imputer.fit_transform(numeric_data)
    data_filled = pd.DataFrame(numeric_data_filled, columns=numeric_data.columns)

    # Feature Scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data_filled)

    # Convert scaled_features to a DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=data_filled.columns)

    return scaled_df


def train_model(X_train, y_train):
    # Using RandomForestRegressor for simplicity
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Grid search for hyperparameter tuning
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    
    # Train model with best parameters
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test):
    # Make predictions
    predictions = model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error:", mse)


def save_model(model, file_path):
    with open(file_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    print("Model saved successfully.")


def print_least_emission_vehicles(model, file_path, num_vehicles=5):
    # Preprocess the data using the provided file path
    scaled_original_df = preprocess_data(file_path)

    # Ensure only relevant features used during training are present
    X_original = scaled_original_df.drop(columns=['CO2 Emissions(g/km)'])

    # Make predictions on the preprocessed dataset
    predictions = model.predict(X_original)

    # Add predicted CO2 emissions to the original dataframe
    original_df = pd.read_csv(file_path)  # Load the original dataset
    original_df['Predicted CO2 Emissions(g/km)'] = predictions

    # Sort the dataframe based on predicted CO2 emissions
    sorted_df = original_df.sort_values(by='Predicted CO2 Emissions(g/km)')

    # Print the top 'num_vehicles' rows
    print(f"\nList of {num_vehicles} vehicles with the least CO2 emissions:")
    print(sorted_df.head(num_vehicles))


def main():
    # Step 1: Preprocess Data
    file_path = 'CO2 Emissions_Canada.csv'

    # Step 2: Splitting Data
    original_df = pd.read_csv(file_path)  # Load the original dataset
    scaled_df = preprocess_data(file_path)  # Preprocess the original dataset

    X = scaled_df.drop(columns=['CO2 Emissions(g/km)'])
    y = scaled_df['CO2 Emissions(g/km)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Train Model
    model = train_model(X_train, y_train)

    # Step 4: Print list of least CO2 emission vehicles and corresponding attributes
    print_least_emission_vehicles(model, file_path)

    # Step 5: Evaluate Model
    evaluate_model(model, X_test, y_test)

    # Step 6: Save Model
    save_model(model, 'co2_emission_model.pkl')


if __name__ == "__main__":
    main()


# In[ ]:





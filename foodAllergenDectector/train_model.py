import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define functions for data loading and preprocessing
def load_data(data_path):
  """
  Loads data from a CSV file and performs basic cleaning.

  Args:
      data_path (str): Path to the CSV file containing the data.

  Returns:
      tuple: A tuple containing the features DataFrame and target Series.
  """
  try:
    data = pd.read_csv(data_path)
    return data.drop('Prediction', axis=1), data['Prediction']
  except FileNotFoundError:
    print(f"Error: File not found at {data_path}")
    return None, None

def preprocess_data(features, target):
  """
  Preprocesses the data by handling missing values and encoding categorical features.

  Args:
      features (pd.DataFrame): DataFrame containing features.
      target (pd.Series): Series containing the target variable.

  Returns:
      tuple: A tuple containing the preprocessed features, encoded features, and scaler object.
  """
  # Check for missing values
  print(f"Number of NaN values in features: {features.isnull().sum().sum()}")
  print(f"Number of NaN values in target: {target.isnull().sum()}")

  # Imputation strategy (can be adjusted based on data analysis)
  features = features.fillna(features.median())  # Replace with median for numerical features
  target = target.fillna(target.mode()[0])  # Replace with most frequent value for target

  # Categorical feature encoding
  encoder = OneHotEncoder()
  encoded_features = encoder.fit_transform(features)

  # Feature scaling (consider if necessary)
  scaler = StandardScaler(with_mean=False)
  scaled_features = scaler.fit_transform(encoded_features)

  return features, encoded_features, scaler

# Define the training function
def train_model(features, target, test_size=0.2, random_state=42):
  """
  Trains a Random Forest Classifier model on the provided data.

  Args:
      features (pd.DataFrame): DataFrame containing preprocessed features.
      target (pd.Series): Series containing the target variable.
      test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.
      random_state (int, optional): Random seed for splitting data. Defaults to 42.

  Returns:
      tuple: A tuple containing the trained model, accuracy score, and classification report.
  """
  X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)

  model = RandomForestClassifier(n_estimators=100, random_state=random_state)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  report = classification_report(y_test, y_pred)

  return model, accuracy, report

# Define function to save the model and components
def save_model(model, scaler, encoder, model_dir='models'):
  """
  Saves the trained model, scaler, and encoder to specified directory.

  Args:
      model (object): Trained model object.
      scaler (object): Fitted scaler object.
      encoder (object): Fitted encoder object.
      model_dir (str, optional): Directory to save the model components. Defaults to 'models'.
  """
  os.makedirs(model_dir, exist_ok=True)
  model_path = os.path.join(model_dir, 'allergen_model.joblib')
  scaler_path = os.path.join(model_dir, 'scaler.joblib')
  encoder_path = os.path.join(model_dir, 'encoder.joblib')

  joblib.dump(model, model_path)
  joblib.dump(scaler, scaler_path)
  joblib.dump(encoder, encoder_path)

# Load data
data_path = 'data/csv/food_ingredients_and_allergens.csv'

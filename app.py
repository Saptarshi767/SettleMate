import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import joblib
import streamlit as st
import re

# Step 1: Load the Dataset
data = pd.read_csv('data.csv')

# Step 2: Data Preprocessing
# Check for missing values
data.fillna(method='ffill', inplace=True)

# Fit LabelEncoders for categorical variables
label_encoder_dispute_type = LabelEncoder()
data['dispute_type'] = label_encoder_dispute_type.fit_transform(data['dispute_type'])

label_encoder_jurisdiction = LabelEncoder()
data['jurisdiction'] = label_encoder_jurisdiction.fit_transform(data['jurisdiction'])

# Remove the encoding for legal_text unless you're using it as a feature
# If you do need to encode legal_text, create a separate LabelEncoder
# label_encoder_legal_text = LabelEncoder()
# data['legal_text'] = label_encoder_legal_text.fit_transform(data['legal_text'])

# Select features and target variable
X = data[['dispute_type', 'jurisdiction', 'amount_in_dispute']]  # Remove legal_text from features
y = data['outcome']

# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Step 8: Save the Model and Encoders
joblib.dump(model, 'arbitrate_model.pkl')
joblib.dump(label_encoder_dispute_type, 'label_encoder_dispute_type.pkl')
joblib.dump(label_encoder_jurisdiction, 'label_encoder_jurisdiction.pkl')

# Streamlit App
st.title('Dispute Resolution Prediction App')

# Load the saved model and label encoders
model = joblib.load('arbitrate_model.pkl')
label_encoder_dispute_type = joblib.load('label_encoder_dispute_type.pkl')
label_encoder_jurisdiction = joblib.load('label_encoder_jurisdiction.pkl')

# Input fields
dispute_type = st.selectbox('Dispute Type', ['Breach', 'Warranty', 'Fraud', 'Contract', 'Payment'])
jurisdiction = st.selectbox('Jurisdiction', ['USA', 'India', 'UK', 'Australia', 'Canada'])
amount_in_dispute = st.number_input('Amount in Dispute', min_value=0)
legal_text = st.text_area('Legal Text', 'Enter the details of the dispute here...')

# Function to extract party names from legal text
def extract_party_names(text):
    pattern = r'(\b[A-Z][a-z]+\b)'  # Matches capitalized words (assumes names are capitalized)
    names = re.findall(pattern, text)
    return names

# Button to make prediction
if st.button('Predict'):
    # Prepare the DataFrame
    new_data = pd.DataFrame({
        'dispute_type': [dispute_type],
        'jurisdiction': [jurisdiction],
        'amount_in_dispute': [amount_in_dispute]
        # legal_text is not included in prediction features
    })

    # Encode categorical variables
    new_data['dispute_type'] = label_encoder_dispute_type.transform(new_data['dispute_type'])
    new_data['jurisdiction'] = label_encoder_jurisdiction.transform(new_data['jurisdiction'])

    # Select features for prediction
    X_new = new_data[['dispute_type', 'jurisdiction', 'amount_in_dispute']]

    # Make predictions
    prediction = model.predict(X_new)

    # Extract party names from the legal text
    party_names = extract_party_names(legal_text)
    
    # Display the prediction and extracted party names
    winner_party = party_names[0] if party_names else "Unknown"
    st.success(f'The predicted outcome is: {int(prediction[0])}')
    st.write(f'Winner Party: {winner_party}')

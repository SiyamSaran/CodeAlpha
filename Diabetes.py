import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
data = pd.read_csv("E:\\diabetes.csv")
data.head(10)


# Replace zero values in specified columns with NaN
columns_with_zero_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns_with_zero_values] = data[columns_with_zero_values].replace(0, np.nan)

# Fill missing values with the median of each column
data.fillna(data.median(), inplace=True)

# Separate features and target variable
X = data.drop('Outcome', axis=1)  
y = data['Outcome']  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Save the trained model to a file
joblib.dump(model, 'diabetes_prediction_model.pkl')

# Function to get user input
def get_user_input():
    pregnancies = 3
    glucose = 110
    blood_pressure = 78
    skin_thickness = 32
    insulin = 62
    bmi = 33.2
    dpf = 0.624
    age = 33
   # Combine the input data into a DataFrame with the same column names as X
    user_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                             columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    return user_data

# Get user input
user_input = get_user_input()

# Predict the outcome using the trained model
prediction = model.predict(user_input)

# Output the prediction
if prediction[0] == 1:
    print("The model predicts that the individual is likely to have diabetes.")
else:
    print("The model predicts that the individual is unlikely to have diabetes.")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

file_path = r"E:\Credit Score Classification Dataset.csv" 
data = pd.read_csv(file_path)
data.head()

data.describe()
data.info()

X = data.drop(columns='Credit Score')
y = data['Credit Score']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
categorical_features = ['Gender', 'Education', 'Marital Status', 'Home Ownership']
numerical_features = ['Age', 'Income', 'Number of Children']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

def predict_credit_score():
    while True:
        print("\nEnter the details to predict the credit score:")
        age = validate_input("Age: ", int)
        gender = validate_input("Gender (Male/Female): ", str, ["Male", "Female"])
        income = validate_input("Income: ", int)
        education = validate_input("Education (High School Diploma/Bachelor's Degree/Master's Degree/Doctorate): ", str, ["High School Diploma", "Bachelor's Degree", "Master's Degree", "Doctorate"])
        marital_status = validate_input("Marital Status (Single/Married): ", str, ["Single", "Married"])
        num_children = validate_input("Number of Children: ", int)
        home_ownership = validate_input("Home Ownership (Owned/Rented): ", str, ["Owned", "Rented"])
        user_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Income': [income],
            'Education': [education],
            'Marital Status': [marital_status],
            'Number of Children': [num_children],
            'Home Ownership': [home_ownership]
        })

        prediction = model.predict(user_data)
        predicted_credit_score = label_encoder.inverse_transform(prediction)
        print(f"\nPredicted Credit Score: {predicted_credit_score[0]}")
        continue_predicting = input("\nDo you want to predict another credit score? (yes/no): ").strip().lower()
        if continue_predicting != 'yes':
            break
predict_credit_score()
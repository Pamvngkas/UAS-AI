import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(BASE_DIR, "bestSelling_games.csv")
df = pd.read_csv(csv_path)

def classify_rating(value):
    if value < 2.5:
        return 'Low'
    elif value < 3.5:
        return 'Medium'
    else:
        return 'High'

df['rating_class'] = df['rating'].apply(classify_rating)

features = [
    'reviews_like_rate',
    'all_reviews_number',
    'price',
    'age_restriction',
    'rating',  
    'difficulty',
    'length',
    'developer',
    'supported_os'
]
target = 'rating_class'

X = df[features]
y = df[target]

numeric_features = ['reviews_like_rate', 'all_reviews_number', 'price', 'age_restriction', 'difficulty', 'length']
categorical_features = ['developer', 'supported_os']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

print("=== Classification Report ===")
print(report)

sample = pd.DataFrame([{
    'reviews_like_rate': 90,
    'all_reviews_number': 10000,
    'price': 14.99,
    'age_restriction': 13,
    'rating': 4.0,
    'difficulty': 3,
    'length': 25,
    'developer': 'Valve',
    'supported_os': 'win'
}])

prediction = model.predict(sample)
print(f"Prediksi kualitas game untuk sample: {prediction[0]}")

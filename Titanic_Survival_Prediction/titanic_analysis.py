import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('Titanic-Dataset.csv')

data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data.drop(columns=['Cabin'], inplace=True)

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

print("Initial Data Information:")
print(data.info())
print("\nData after Preprocessing:")
print(data.info())
print("\nData after Converting Categorical Variables:")
print(data.info())

X = data.drop(columns=['Survived'])
y = data['Survived']

X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)
y = y[X.index]

print("\nFeature matrix (X) shape after handling NaNs:", X.shape)
print("Target vector (y) shape after handling NaNs:", y.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy:.2f}')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

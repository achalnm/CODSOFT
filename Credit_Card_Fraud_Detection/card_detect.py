#output might take anywhere between 2 to 8 minutes depending on system specifications

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

data = pd.read_csv('creditcard.csv')

print("Initial Data Information:")
print(data.info())

X = data.drop(columns=['Class'])  
y = data['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)  
model.fit(X_resampled, y_resampled)

y_pred = model.predict(X_test)

print(f'\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
print(f'\nClassification Report:\n{classification_report(y_test, y_pred)}')

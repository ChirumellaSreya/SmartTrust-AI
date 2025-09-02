import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
data = pd.read_csv('transactions.csv')

# Drop columns that are not useful as features
X = data.drop(['label', 'transaction_id', 'user_id', 'date_time', 'ip_address'], axis=1)

# Convert categorical columns to numeric using one-hot encoding
X = pd.get_dummies(X, columns=['merchant_type', 'location', 'device_type', 'channel'])

# Target variable
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Print classification report to evaluate model
print(classification_report(y_test, y_pred))


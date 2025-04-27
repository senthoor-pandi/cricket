# model/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
df = pd.read_csv('../dataset/matches.csv')

# Select important columns
df = df[['team1', 'team2', 'venue', 'winner']]

# Drop rows with missing values
df = df.dropna()

# Prepare features and target
X = df[['team1', 'team2', 'venue']]
y = df['winner']

# Convert text columns to numbers (Label Encoding)
from sklearn.preprocessing import LabelEncoder
le_team = LabelEncoder()
le_venue = LabelEncoder()

X['team1'] = le_team.fit_transform(X['team1'])
X['team2'] = le_team.transform(X['team2'])
X['venue'] = le_venue.fit_transform(X['venue'])
y = le_team.transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model and encoders
with open('../model/cricket_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('../model/team_encoder.pkl', 'wb') as f:
    pickle.dump(le_team, f)

with open('../model/venue_encoder.pkl', 'wb') as f:
    pickle.dump(le_venue, f)

print("✅ Model trained and saved successfully!")

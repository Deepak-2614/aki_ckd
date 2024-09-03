import google.generativeai as gemini
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pickle

# Set up the Gemini API key
gemini.configure(api_key="AIzaSyBi5Rs6lCU_XjW2QIyYW-oDf-ldm-NLZAw")

# Load and preprocess the dataset
df = pd.read_csv("C:/Users/Leka/Downloads/Kidneyyyy/Ckd & Aki/kidney_disease_classification_updated.csv")

# Label Encoding for categorical variables
lb = LabelEncoder()
categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'GFR', 'classification']
for col in categorical_columns:
    df[col] = lb.fit_transform(df[col])

# Replace '\t?' with NaN and convert the relevant columns to float
df.replace('\t?', np.nan, inplace=True)
columns_to_convert = ['bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 
                      'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 
                      'cad', 'appet', 'pe', 'ane', 'GFR', 'classification']

for column in columns_to_convert:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Fill missing values with the mean of the respective columns
df.fillna(df.mean(), inplace=True)

# Split the data into features and target
X = df.drop('classification', axis=1)
y = df['classification']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
knn = KNeighborsClassifier(n_neighbors=2)
rf = RandomForestClassifier()
svm = SVC(probability=True)
gb = GradientBoostingClassifier()
nb = GaussianNB()

# Create a Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('knn', knn),
    ('rf', rf),
    ('svm', svm),
    ('gb', gb),
    ('nb', nb)
], voting='soft')

# Train the Voting Classifier
voting_clf.fit(X_train, y_train)

# Save the model to a pickle file
with open('Ckd and Aki/kidney_disease_voting_model.pkl', 'wb') as model_file:
    pickle.dump(voting_clf, model_file)    

# Save the LabelEncoder as well
with open('Ckd and Aki/label_encoder.pkl', 'wb') as le_file:
    pickle.dump(lb, le_file)

print("Model and LabelEncoder have been saved in pickle format.")
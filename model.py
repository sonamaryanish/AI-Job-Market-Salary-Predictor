import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

# Load CSV file
df = pd.read_csv("data/ai_job_dataset.csv")

# remove missing values
df = df.dropna()

# Feature Engineering
df.drop(columns=['job_id','salary_currency','employee_residence','posting_date','application_deadline','job_description_length','benefits_score','company_name'], inplace=True)
print("\n Columns are ",df.columns)


#Removing outliers
Q1 = df['salary_usd'].quantile(0.25)
Q3 = df['salary_usd'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_clean = df[(df['salary_usd'] >= lower_bound) &
              (df['salary_usd'] <= upper_bound)]


categorical_columns = df_clean.select_dtypes(include=['object']).columns
print(categorical_columns)

encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    encoders[col] = le

with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

X = df_clean.drop("salary_usd", axis=1)
y = df_clean["salary_usd"]

print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Coverting to Dataframe
x=pd.DataFrame(X_scaled)

with open('scaling.pkl', 'wb') as f:
    pickle.dump(scaler, f)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



# RandomForest Regression
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# save model
pickle.dump(model, open("salary_model.pkl", "wb"))






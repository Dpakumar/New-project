import pandas as pd
import sklearn 
import numpy as np

#LOAD THE SAMPLE DATASET
data = "https://raw.githubusercontent.com/Dpakumar/New-project/main/Car2DB_eng_cut.csv"
df = pd.read_csv(data)

#DATA CLEANING
#Drop unnessory columns
df.drop(columns=['id_trim','Generation','Trim','Cargo compartment volume [m3]','Min trunk capacity [litre]',
                 'Serie','Cargo compartment (Length x Width x Height) [mm]','Loading height [mm]',
                 'Front/rear axle load [kg]','Permitted road-train weight [kg]','Front track [mm]','Stroke cycle [mm]',
                 'Cylinder bore [mm]','Presence of intercooler','Boost type','Valves per cylinder','Injection type',
                 'Cylinder layout','Turnover of maximum torque [RPM]','Payload [kg]','Unnamed: 58','Max power at RPM [RPM]'], inplace=True)

df.describe()  # to know more about given data

#Label Encoding
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cols=['Make','Model','Body type','Engine type','Rear brakes','Back suspension','Front suspension','Front brakes','Gearbox type',
      'Drive wheels','Cruising range [km]','Fuel','Emission standards']
df[cols] = df[cols].apply(le.fit_transform)
print(df)

#calculating the null values
#print(df.isnull().sum())
#drop the null values
#df.dropna(inplace=True)
#or
#Replacing the null values with respective columns mean value
cols1 = ['Rear track [mm]', 'Curb weight [kg]', 'Max trunk capacity [litre]', 'Full weight [kg]',
                'Wheelbase [mm]', 'Ground clearance [mm]', 'Maximum torque [Nm]', 'Engine power [bhp]',
                'Capacity [cm3]', 'Number of cylinders', 'Number of gear', 'Turning circle [m]',
                'Cruising range [km]', 'Fuel tank capacity [litre]', 'Acceleration (0-100 km/h) [second]',
                'Max speed [km/h]', 'City driving fuel consumption per 100 km [litre]',
                'Highway driving fuel consumption per 100 km [litre]',
                'Mixed driving fuel consumption per 100 km [litre]']
df[cols1] = df[cols1].fillna(df[cols1].mean())


print(df.isnull().sum()) # Checking the null values
print(df)
#Things can to do with this data
#Outlier detection
#Finding the IQR
percentile25 = df['Mixed driving fuel consumption per 100 km [litre]'].quantile(0.25)
percentile75 = df['Mixed driving fuel consumption per 100 km [litre]'].quantile(0.75)
iqr=percentile75 - percentile25
print(iqr)
print(percentile25)
print(percentile75)

#Finding upper and lower limit
upper_limit = percentile75 + 1 * iqr
lower_limit = percentile25 - 1 * iqr
print(upper_limit)
print(lower_limit)

#finding outliers
df[df['Mixed driving fuel consumption per 100 km [litre]'] > upper_limit]
df[df['Mixed driving fuel consumption per 100 km [litre]'] < lower_limit]

#trimming
new_df1 = df[df['Mixed driving fuel consumption per 100 km [litre]'] < upper_limit]
new_df = new_df1[new_df1['Mixed driving fuel consumption per 100 km [litre]'] > lower_limit]
print(new_df)

# Data Visualization 
import matplotlib.pyplot as plt
import seaborn as sns
#sns.pairplot(df[cols1])
#print(plt.show())

# data Analysis 
correlation_matrix = df[cols1].corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.4)
plt.show()


#Insights from the data
#which one have better mileage in(ex)

from itertools import groupby
new_df_group=df[['Mixed driving fuel consumption per 100 km [litre]', 'Highway driving fuel consumption per 100 km [litre]',
                 'City driving fuel consumption per 100 km [litre]']]
a=new_df.groupby(['Mixed driving fuel consumption per 100 km [litre]'], as_index=False).mean()
print(a)

# Model Building and error evaluation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
X = df.drop(columns=['Mixed driving fuel consumption per 100 km [litre]'])
y = df['Mixed driving fuel consumption per 100 km [litre]']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)  

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")











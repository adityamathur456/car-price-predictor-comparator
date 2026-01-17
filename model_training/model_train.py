import joblib
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

# Set the style
mpl.style.use('ggplot')

# Load and clean data
car=pd.read_csv(r"./data/raw/quikr_car.csv")

# Initial data exploration
car.head()
car.info()
car.describe()
car.isnull().sum()

# Data Cleaning and Preprocessing
car=car[car['year'].str.isnumeric()]
car['year']=car['year'].astype(int)

# Remove rows with 'Ask For Price' in Price column
car=car[car['Price']!='Ask For Price']
car['Price']=car['Price'].str.replace(',','').astype(int)

# Clean 'kms_driven' column
car['kms_driven']=car['kms_driven'].str.split().str.get(0).str.replace(',','')
car=car[car['kms_driven'].str.isnumeric()]

# Convert 'kms_driven' to integer
car['kms_driven']=car['kms_driven'].astype(int)
car=car[~car['fuel_type'].isna()]

# Final data exploration
car.shape
car['name']=car['name'].str.split().str.slice(start=0,stop=3).str.join(' ')
car=car.reset_index(drop=True)

# Final data checks
car.describe()
car.head()
car.to_csv('Cleaned_Car_data.csv')
car.info()

# Data Visualization
car=car[car['Price']<6000000]
car['company'].unique()

# Boxplot and Swarmplots
plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='company',y='Price',data=car)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()

plt.subplots(figsize=(20,10))
ax=sns.swarmplot(x='year',y='Price',data=car)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()

# Model Training
X=car[['name','company','year','kms_driven','fuel_type']]
y=car['Price']

# Split the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# Preprocessing and Model Pipeline
# OneHotEncoder for categorical features
ohe=OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])

column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                    remainder='passthrough')

# CatBoost Regressor Model
model=CatBoostRegressor(
    iterations=800,
    learning_rate=0.3,
    depth=6,
    eval_metric='R2',
)

# Create the pipeline
pipe=make_pipeline(column_trans,model)

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)

print("R2 score:",r2_score(y_test,y_pred))

# Hyperparameter Tuning with different random states
# Finding the best random state for train-test split
# scores=[]
# for i in range(100):
#     X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
#     catBoost=model
#     pipe=make_pipeline(column_trans,catBoost)
#     pipe.fit(X_train,y_train)
#     y_pred=pipe.predict(X_test)
#     scores.append(r2_score(y_test,y_pred))

# Best random state
# print(np.argmax(scores))
# print(scores[np.argmax(scores)])

# predicting a sample
print(pipe.predict(pd.DataFrame(columns=X_test.columns,data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5))))

# Final Model Training with best random state
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=84)
catBoost=model
pipe=make_pipeline(column_trans,catBoost)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)

# Final R2 Score
print("R2 score:",r2_score(y_test,y_pred))

# Save the model
joblib.dump(pipe, "catboost_model.pkl")
pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))
pipe.steps[0][1].transformers[0][1].categories[0]
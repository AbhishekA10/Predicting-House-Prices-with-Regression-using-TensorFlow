#Importing libraries
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#Importing dataset 
data=pd.read_csv('kc_house_data.csv')

#Splitting the training and testing data
X=data.drop(['price','id','date','zipcode'],axis=1).values
y=data['price'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)

#Scaling features using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#Creating Sequential model
model=Sequential()
model.add(Dense(17,activation='relu'))
model.add(Dense(17,activation='relu'))
model.add(Dense(17,activation='relu'))
model.add(Dense(17,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

#Training the model
model.fit(x=X_train,y=y_train,epochs=200,validation_data=(X_test,y_test),callbacks=[EarlyStopping(monitor='val_loss',patience=5)])

#Predicting output
p=model.predict(X_test)

#Printing explained_variance_score
from sklearn.metrics import explained_variance_score
explained_variance_score(y_test,p)
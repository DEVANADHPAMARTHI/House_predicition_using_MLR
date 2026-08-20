import numpy as np
import pandas as pd
import sklearn
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
class HOUSE_PREDICTOR:
    reg=LinearRegression()
    def __init__(self,dfs):
        self.X=df.iloc[ : ,1:]
        self.y=df.iloc[ : ,0]
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=42)
    def train(self):
        self.reg.fit(self.X_train,self.y_train)
    def accuracy(self):
        self.train_predict_values=self.reg.predict(self.X_train)
        self.test_predict_values=self.reg.predict(self.X_test)
        train_num,test_num=0,0
        train_den,test_den=0,0
        for j,k in zip(self.y_train,self.train_predict_values):
            train_num=train_num+(k-j)**2
            train_den=train_den+(j-np.mean(self.y_train))**2
        print("The model accuracy with train data",1-(train_num/train_den))
        for z,c in zip(self.y_test,self.test_predict_values):
            test_num=test_num+(z-c)**2
            test_den=test_den+(z-np.mean(self.y_test))**2
        print("The model accuracy with test data",1-(test_num/test_den))
    def loss(self):
        train_loss,test_loss=0,0
        for j,k in zip(self.y_train,self.test_predict_values):
            test_loss=test_loss+(j-k)**2
        print("The model loss with test data",np.sqrt(test_loss/len(self.y_test)))
        for j,k in zip(self.y_train,self.test_predict_values):
            train_loss=train_loss+(j-k)**2
        print("The model loss with train data",np.sqrt(train_loss/len(self.y_train)))
df=pd.read_csv("data.csv")
df['date']=pd.to_datetime(df['date'])
df['year']=df['date'].dt.year
df['month']=df['date'].dt.month
df['day']=df['date'].dt.day
cities=df['city'].unique()
city_map={}
for i,city in enumerate(cities):
    city_map[city]=i
df['city']=df['city'].map(city_map)
df['country']=0
df=df.drop(['date'],axis=1)
obj=HOUSE_PREDICTOR(df)
obj.train()
obj.accuracy()
obj.loss()
with open("MLResults.pkl","wb") as f:
    pickle.dump(obj.reg, f)

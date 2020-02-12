# import needed library
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#load iris dataset
data_iris = load_iris()
iris_X, iris_y = load_iris(return_X_y=True)
feature_iris = data_iris['feature_names']

#load play tennis dataset
play_tennis =  pd.read_csv('play_tennis.csv')
play_tennis_X = play_tennis.drop('play',axis=1)
play_tennis_y = play_tennis['play']

#transform iris into dataframe
iris_X=pd.DataFrame(iris_X)
iris_y=pd.DataFrame(iris_y)

#create index so be merge
iris_X=iris_X.reset_index()
iris_y=iris_y.reset_index()

iris_y.rename(columns = {0:4}, inplace = True) 

#merge dataset iris
iris=iris_X.merge(iris_y)

#drop index
iris.drop("index",axis=1,inplace=True)

#rename iris columns 
iris.rename(columns = {0:feature_iris[0],1:feature_iris[1],2:feature_iris[2],3:feature_iris[3],4:"target"}, inplace = True)

#save column name into play_tennis_column
play_tennis_column = ['day', 'outlook', 'temp', 'humidity', 'wind','play']

#label encode to encode play tennis categorical data
for col in play_tennis_column:
    lbl = LabelEncoder()
    lbl.fit(list(play_tennis[col].values))
    play_tennis[col] =lbl.transform(list(play_tennis[col].values))
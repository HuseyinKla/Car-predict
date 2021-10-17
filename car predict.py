import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf

df = pd.read_excel("veriler/merc.xlsx")

df = df.sort_values("price",ascending=False).iloc[131:]
df = df.drop("transmission",axis=1)

y = df["price"].values
x = df.drop("price",axis=1).values




from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=10)
print(len(x_test))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.models import Sequential
#from tensorflow.keras.models import Dense
from keras.layers.core import Dense

model = Sequential()

model.add(Dense(12,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(12,activation='relu'))

model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")


model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),batch_size=250,epochs=300)




kayipVeri = pd.DataFrame(model.history.history)
print(kayipVeri.head())

kayipVeri.plot()
plt.show()
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from keras.models import load_model
import streamlit as st 
import yfinance as yf

st.title('Stock Price Prediction of next 5 days')

#user_input
stock_symbol=st.text_input("Please enter the correct ticker name of the stock:")



#last 5 years data with interval of 1 day
data = yf.download(tickers=stock_symbol,period='5y',interval='1d')

data.tail()
st.subheader('Description of the data')
st.write(data.describe()) 

st.subheader('last 5 datastamps')

st.write(data.tail())
st.subheader('opening price')
fig=plt.figure(figsize=(12,6))
plt.plot(data.Open)
plt.title('open')
st.pyplot(fig)



opn = data[['Open']]

ds = opn.values

#st.subheader('TESTING')
#opn = data[['Open']]
#fig=plt.figure(figsize=(12,6))
#plt.plot(opn.values,'r')
#st.pyplot(fig)


import numpy as np
from sklearn.preprocessing import MinMaxScaler

normalizer = MinMaxScaler(feature_range=(0,1))
ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))


train_size = int(len(ds_scaled)*0.70)
test_size = len(ds_scaled) - train_size


ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]

def create_ds(dataset,step):
    Xtrain, Ytrain = [], []
    for i in range(len(dataset)-step-1):
        a = dataset[i:(i+step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i + step, 0])
    return np.array(Xtrain), np.array(Ytrain)



time_stamp = 100
X_train, y_train = create_ds(ds_train,time_stamp)
X_test, y_test = create_ds(ds_test,time_stamp)


#Reshaping data to fit into LSTM model
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


from keras.models import Sequential
from keras.layers import Dense, LSTM

model=load_model('keras_model')





train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = normalizer.inverse_transform(train_predict)
test_predict = normalizer.inverse_transform(test_predict)

st.subheader('Compairing Values')
fig=plt.figure(figsize=(12,6))
plt.plot(normalizer.inverse_transform(ds_scaled))
plt.plot(train_predict)
plt.plot(test_predict)
plt.title('Blue Line is the actual data . Orange line is the predicted value of train data. Green line is the predicted value of the test data')
st.pyplot(fig)



st.subheader('Combining the predited data to create uniform data visualization')
fig=plt.figure(figsize=(12,6))
test = np.vstack((train_predict,test_predict))
plt.plot(normalizer.inverse_transform(ds_scaled))
plt.plot(test)
st.pyplot(fig)


a=len(ds_test)

b=a-100

fut_inp = ds_test[b:]
fut_inp = fut_inp.reshape(1,-1)

tmp_inp = list(fut_inp)

#Creating list of the last 100 data
tmp_inp = tmp_inp[0].tolist()

#Predicting next 30 days price suing the current data
#It will predict in sliding window manner (algorithm) with stride 1
lst_output=[]
n_steps=100
i=0
while(i<5):
    
    if(len(tmp_inp)>100):
        fut_inp = np.array(tmp_inp[1:])
        fut_inp=fut_inp.reshape(1,-1)
        fut_inp = fut_inp.reshape((1, n_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        tmp_inp = tmp_inp[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp = fut_inp.reshape((1, n_steps,1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1
    




lenth_of_ds_scaled=len(ds_scaled)-100

#Creating a dummy plane to plot graph one after another
plot_new=np.arange(1,101)
plot_pred=np.arange(101,106)


st.subheader('Creating a dummy plane to plot graph one after another')
fig=plt.figure(figsize=(12,6))
plt.plot(plot_new, normalizer.inverse_transform(ds_scaled[lenth_of_ds_scaled:]))
plt.plot(plot_pred, normalizer.inverse_transform(lst_output))
plt.title('Orange line is the predicted values')
st.pyplot(fig)


ds_new = ds_scaled.tolist()


st.subheader('Entends helps us to fill the missing value with approx value')
fig=plt.figure(figsize=(12,6))
ds_new.extend(lst_output)
plt.plot(ds_new[1200:])
st.pyplot(fig)



final_graph = normalizer.inverse_transform(ds_new).tolist()


st.subheader('Next 5 days open')
fig=plt.figure(figsize=(12,6))
plt.plot(final_graph,)
plt.ylabel("Price")
plt.xlabel("Time")
plt.title("{0} prediction of next 5 days open".format(stock_symbol))
plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'NEXT 5D: {0}'.format(round(float(*final_graph[len(final_graph)-1]),2)))
plt.legend()
st.pyplot(fig)

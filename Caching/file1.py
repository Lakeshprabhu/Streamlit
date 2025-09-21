import streamlit as st
import time
import numpy as np
from sklearn.linear_model import LinearRegression


st.write('Caching Test')

st.button('Test Caching')

st.subheader('Cache Data')

@st.cache_data
def testing():
    time.sleep(2)

    return "done"

out = testing()
st.write(out)


st.subheader('Cache Resource')


def Linear_model(predo):
    time.sleep(2)
    X = np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
    Y = np.array([1,2,3,4,5,6,7,8])
    model = LinearRegression()
    model = model.fit(X,Y)
    pred = model.predict(predo.reshape(-1,1))
    st.write(pred)
    return pred


aba = (Linear_model(9))

st.write(aba)

print(aba)

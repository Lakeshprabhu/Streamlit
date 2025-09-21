import streamlit as st
import pandas as pd

with st.sidebar:
    st.write("Pop")

col1,col2,col3 = st.columns(3)

col1.write("abc")

slo = col2.slider('Choose a slider',max_value=10,min_value=0,value=1)

col3.write(slo)

df = pd.read_csv("halo.csv")

tab1 , tab2 = st.tabs(['Line plot','Bar plot'])

with tab1:
    tab1.write('A line plot')
    st.line_chart(df,x='year',y=['col1','col2','col3'])

with tab2:
    tab2.write("A bar plot")
    st.bar_chart(df,x='year',y=['col1','col2','col3'])


with st.expander("click to open"):
    st.write("abo")

with st.container():
    st.write("This is inside a container")

st.write("This is outside")
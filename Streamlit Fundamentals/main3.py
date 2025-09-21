import streamlit as st
import pandas as pd


primary_btn = st.button(label='Primary',type='primary')
secondary_btn = st.button(label='Secondary',type='secondary')


if primary_btn:
    st.write("Primary Button chosen")
if secondary_btn:
    st.write("Secondary Button chosen")


st.divider()

check_box = st.checkbox("Remember Me")

if check_box:
    st.write("r")
else:
    st.write("f")


st.divider()

df = pd.read_csv('halo.csv')
radio = st.radio('chose a column',options = df.columns[1:],index = 0 , horizontal=False)
st.write(radio)

st.divider()

select_box = st.selectbox("chose a column",options = df.columns[1:],index=0)
st.write(select_box)

st.divider()

multi = st.multiselect('chose as many cols',options = df.columns[1:],default=['col1'],max_selections=3)
st.write(*multi)



st.divider()

slider = st.slider('Pick a number',max_value=10,min_value=0,value=0,step = 1)
st.write(slider)

st.divider()

text_input = st.text_input('Enter a name',placeholder="Johnny")
st.write(text_input)

st.divider()

num_input = st.number_input("Enter a number",max_value=10,min_value=0,value=0,step=1)
st.write(num_input)

st.divider()

text_box = st.text_area("write",height=300,placeholder='job')
st.write(text_box)
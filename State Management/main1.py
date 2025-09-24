import streamlit as st


st.title("Exercise: State Management")

st.header("Temperature Conversion")


if 'cel'not in st.session_state:
    st.session_state['cel'] = 0.0

if 'far'not in st.session_state:
    st.session_state['far'] = 32.00

if 'kel'not in st.session_state:
    st.session_state['kel'] = 273.15

col1,col2,col3 = st.columns(3)

def change_cel():
    celsius = st.session_state['cel']

    st.session_state['far'] = (celsius * 1.8) + 32
    st.session_state['kel'] = (celsius + 273.15)



def change_far():
    farenheit = st.session_state['far']

    st.session_state['cel'] = (farenheit * 1.8) - 32
    st.session_state['kel'] = (farenheit - 32) * 1.8 + 273.15



def change_kel():
    kelvin = st.session_state['kel']

    st.session_state['cel'] = (kelvin) - 273.15
    st.session_state['far'] = ((kelvin - 273.15) * 1.8) + 32


def change_all(cel,far,kel):
    st.session_state['cel'] = cel
    st.session_state['far'] = far
    st.session_state['kel'] = kel



with col1:
    st.number_input('CEL',step=0.01,key='cel',on_change=change_cel)
    st.number_input(label='Add to Celsius')

with col2:
    st.number_input('Farenheit',step=0.01,key = "far" ,on_change=change_far)

with col3:
    st.number_input('Kelvin', step=0.01, key = "kel",on_change=change_kel)





abc1,abc2,abc3 = st.columns(3)

with abc1:
    st.button("🧊 Freezing point of water",on_click=change_all,kwargs=dict(cel=0.00,far=32,kel=273.15))

with abc2:
    st.button("🔥 Boiling point of water",on_click=change_all,kwargs=dict(cel=100.00,far=212.00,kel=373.15))


with abc3:
    st.button("🥶 Absolute point of water",on_click=change_all,kwargs=dict(cel= 273.15,far=-459.67,kel=0.00))



st.write(st.session_state)
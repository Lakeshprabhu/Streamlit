import streamlit as st


with st.form('my_form'):
    st.text("What would you like to order:")


    ao = st.selectbox(label='Appetizers',options=('choice1','choice2','choice3'))


    bo = st.selectbox(label='Main Course',options=('choice1','choice2','choice3'))


    co = st.selectbox(label='Dessert',options=('choice1','choice2','choice3'))

    che = st.checkbox(label='Are you bringing your wine')

    do = st.date_input(label='When are you coming')

    ti = st.time_input(label='At what are you coming ?')

    top = st.text_area(label='Any allergies',placeholder='Peanuts')

    submitted = st.form_submit_button("Submit")

    if submitted:
        st.write(ao,bo,co,che,do,ti,top)
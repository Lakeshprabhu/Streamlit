import streamlit as st


st.title('Title')

st.header("Head")

st.subheader("Body")

st.markdown("**ty**")
st.markdown("# home")
st.markdown("## dome")

st.caption("This is a caption")

st.code("""import pandas as pd
        pd.read()""")

st.caption("This is a caption")

st.text("This is a text")

st.latex("x^2 = yx")

st.text("Text above divider")
st.divider()
st.text("Text below divider")

st.write("text")
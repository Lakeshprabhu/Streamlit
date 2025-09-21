import streamlit
import pandas

df = pandas.read_csv("halo.csv")

streamlit.dataframe(df)
streamlit.write(df)

streamlit.table(df)

streamlit.metric(label='Population',value=200,delta=20,delta_color='normal')

streamlit.divider()

streamlit.line_chart(df,x='year',y=['col1','col2'])

streamlit.divider()

streamlit.area_chart(df,x='year',y=['col1','col2'])


streamlit.divider()

streamlit.bar_chart(df,x='year',y=['col1','col2'])
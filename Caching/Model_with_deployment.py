import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier

URL = "https://raw.githubusercontent.com/marcopeix/MachineLearningModelDeploymentwithStreamlit/master/17_caching_capstone/data/mushrooms.csv"
COLS = ['class', 'odor', 'gill-size', 'gill-color', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'ring-type', 'spore-print-color']


@st.cache_data
def read_csv(url,cols):
    df = pd.read_csv(url)
    df = df[cols]

    return df


@st.cache_resource
def lb_encoding(data):
    le =LabelEncoder()
    le = le.fit(data['class'])
    return le

@st.cache_resource
def or_encoding(data):
    o = OrdinalEncoder()
    o = o.fit(data.columns[1:])
    return o

@st.cache_resource(show_spinner="encoding data....")
def perform_encoding(data,_xencode,_yencode):
    data['class'] = _xencode.transform(data['class'])
    x_cols = data.columns[1:]
    data[x_cols] = _yencode.transform(data['x_cols'])

    return data

@st.cache_resource(show_spinner='Training model ...')
def train_model(data):
    X = data.drop(['class'],axis=1)
    y = data['class']

    model = GradientBoostingClassifier(max_depth = 5 , random_state = 42)
    model = model.fit(X,y)

    return model



if __name__ == "__main__":

    df = read_csv(URL,COLS)
    st.title("Mushroom Classifier")

    col1,col2,col3 = st.columns(3)

    with col1:
        odor = st.selectbox('Odor',('a - almond', 'l - anisel', 'c - creosote', 'y - fishy', 'f - foul', 'm - musty', 'n - none', 'p - pungent', 's - spicy'))
        stalk_surface_above_ring = st.selectbox('Stalk surface above ring',
                                                ('f - fibrous', 'y - scaly', 'k - silky', 's - smooth'))
        stalk_color_below_ring = st.selectbox('Stalk color below ring',
                                              ('n - brown', 'b - buff', 'c - cinnamon', 'g - gray', 'o - orange',
                                               'p - pink', 'e - red', 'w - white', 'y - yellow'))
    with col2:
        gill_size = st.selectbox('Gill size', ('b - broad', 'n - narrow'))
        stalk_surface_below_ring = st.selectbox('Stalk surface below ring',
                                                ('f - fibrous', 'y - scaly', 'k - silky', 's - smooth'))
        ring_type = st.selectbox('Ring type', ('e - evanescente', 'f - flaring', 'l - large', 'n - none', 'p - pendant',
                                               's - sheathing', 'z - zone'))
    with col3:
        gill_color = st.selectbox('Gill color',
                                  ('k - black', 'n - brown', 'b - buff', 'h - chocolate', 'g - gray', 'r - green',
                                   'o - orange', 'p - pink', 'u - purple', 'e - red', 'w - white', 'y - yellow'))
        stalk_color_above_ring = st.selectbox('Stalk color above ring',
                                              ('n - brown', 'b - buff', 'c - cinnamon', 'g - gray', 'o - orange',
                                               'p - pink', 'e - red', 'w - white', 'y - yellow'))
        spore_print_color = st.selectbox('Spore print color',
                                         ('k - black', 'n - brown', 'b - buff', 'h - chocolate', 'r - green',
                                          'o - orange', 'u - purple', 'w - white', 'y - yellow'))

    st.subheader('Step 2 : Prediction')

    st.button("Predict",type='primary')
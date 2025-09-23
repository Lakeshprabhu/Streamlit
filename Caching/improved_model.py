import joblib
import streamlit as st
import pandas as pd
from joblib import dump
import numpy as np
from pandas.core.common import random_state
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from joblib import load
URL = "https://raw.githubusercontent.com/marcopeix/MachineLearningModelDeploymentwithStreamlit/refs/heads/master/18_caching_capstone/data/mushrooms.csv"
COLS = ['class', 'odor', 'gill-size', 'gill-color', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'ring-type', 'spore-print-color']
#
#
# df = pd.read_csv(URL)
# df = df[COLS]
#
# pipe = Pipeline([
#     ('encoder',OrdinalEncoder()),
#     ('gbc',GradientBoostingClassifier(max_depth=5,random_state=42))
#
# ])
#
# X = df.drop(['class'],axis=1)
# y = df['class']
#
# pipe.fit(X,y)
#
# dump(pipe,"model/pipe.joblib")


@st.cache_resource(show_spinner='Loading Model....')
def train_model():
    mo = joblib.load('model/pipe.joblib')
    return mo

@st.cache_resource(show_spinner='Predicting....')
def predict_model(_mo,y_goal):
    features = [i[0] for i in y_goal]
    features = np.array(features).reshape(1,-1)
    pred = _mo.predict(features)
    return pred

if __name__ == "__main__":




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

    po = st.button("Predict",type='primary')

    if po:
        y_goal = [odor,
                  gill_size,
                  gill_color,
                  stalk_surface_above_ring,
                  stalk_surface_below_ring,
                  stalk_color_above_ring,
                  stalk_color_below_ring,
                  ring_type,
                  spore_print_color]

        mo = train_model()
        pred = predict_model(mo,y_goal)

        if pred == "e":
            st.write("Edible")

        else:
            st.write("poisonous")



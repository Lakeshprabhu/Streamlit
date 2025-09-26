from joblib import dump
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectPercentile,mutual_info_regression

URL = "honda_toyota_ca.csv"
df = pd.read_csv(URL)

X = (df[['miles','year','make','model','engine_size','state']])
y = df['price']

x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=df[['make','model']],test_size = 0.2,shuffle=True,random_state=42)

cat_index = [3,4,5]

cat_pipe = Pipeline(
    steps = [("encoder",OrdinalEncoder()),
             ('selector',SelectPercentile(mutual_info_regression,percentile=50))

    ]
)

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

preprocessor = ColumnTransformer(
    transformers=[
        ('cat',cat_pipe,cat_index),

    ]
)


model = Pipeline(
    steps = [
        ('preprocessor',preprocessor),
        ('regressor',GradientBoostingRegressor(random_state=42))
    ]
)

model.fit(x_train,y_train)
print(x_train.columns)

print(model.score(x_test,y_test))

dump(model,'model/model.joblib')
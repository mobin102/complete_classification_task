import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

# this useful fuction that I write
from utils import cat_num_split, null_ratio, sample_imputer

df_test= pd.read_csv("../data/data_test_users.csv", na_values="?", header=None)
#split categorical and numerical columns
cat_df_test, num_df_test = cat_num_split(df_test)

#filter numerical columns which contain more than half percent of nullity
not_null_columns, _ = null_ratio(num_df_test,atmost_null=0.5)

# load informations and models
selected_cloumns = load("../model/selected_cloumns.joblib")
transformer = load("../model/transformer.joblib")
final_model = load("../model/final_model.joblib")

#integrate the columns
final_cloumns = selected_cloumns+not_null_columns + list(cat_df_test.columns)

#fill the null values
df_test= sample_imputer(df_test,final_cloumns)
imputed_df = df_test[final_cloumns].copy()
imputed_df_transformed = transformer.transform(imputed_df)

#hard classification on test data
predication = final_model.predict(imputed_df_transformed)

#To save in data directory
pd.DataFrame(data=predication,
            index=df_test.index,
            columns=["predication"]).to_csv("../data/predication.csv")
#%%
import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

models = mlflow.search_registered_models(filter_string="name='model_churn'")
latest_version = max([i.version for i in models[0].latest_versions])

model = mlflow.sklearn.load_model(f"models:/model_churn/{latest_version}")
features = model.feature_names_in_
#%%
model
features

#%%
df = pd.read_csv("../data/abt_churn.csv")
amostra = df[df['dtRef'] == df['dtRef'].max()].sample(3)
amostra = amostra.drop('flagChurn', axis=1)

#%%
predict = model_df['model'].predict_proba(amostra[model_df['features']])[:,1]
amostra['% Churn'] = predict*100
amostra
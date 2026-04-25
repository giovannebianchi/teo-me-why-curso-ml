#%%
import pandas as pd
from sklearn import model_selection, tree, linear_model, naive_bayes, ensemble, metrics, pipeline
import matplotlib.pyplot as plt
from feature_engine import discretisation, encoding
import mlflow

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment(experiment_id='1')

pd.options.display.max_columns = 500
pd.options.display.max_rows = 500

df = pd.read_csv("../data/abt_churn.csv")
df.head()

#%%
oot = df[df["dtRef"] == df['dtRef'].max()].copy()
oot

#%%
df_train = df[df["dtRef"] < df['dtRef'].max()].copy()
df_train["dtRef"]

#%%
features = df_train.columns[2: -1]
target = 'flagChurn'

X, y = df_train[features], df_train[target]

#%%
# SAMPLE - (S)EMMA
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    random_state=42,
                                                                    test_size=0.2,
                                                                    stratify=y)

#%%
print(f"Taxa variável resposta treino: {y_train.mean()}")
print(f"Taxa variável resposta teste: {y_test.mean()}")

#%%
# EXPLORE - S(E)MMA
X_train.isna().sum().sort_values(ascending = False)

#%%
df_analise = X_train.copy()
df_analise[target] = y_train
summary = df_analise.groupby(by=target).agg(['mean', 'median']).T
summary

#%%
summary['diff_abs'] = summary[0] - summary[1]
summary['diff_rel'] = summary[0] / summary[1]
summary.sort_values(by=['diff_rel'], ascending=False)

#%%
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)

feature_importances = (pd.Series(arvore.feature_importances_, index=X_train.columns)
                       .sort_values(ascending=False)
                       .reset_index())
feature_importances['acum.'] = feature_importances[0].cumsum()
feature_importances[feature_importances['acum.'] < 0.96]

#%%
best_features = (feature_importances[feature_importances['acum.'] < 0.96]['index'].tolist())
best_features

#%%
# MODIFY

#Discretizar
tree_discretization = discretisation.DecisionTreeDiscretiser(
  variables=best_features,
  regression=False,
  bin_output='bin_number',
  cv=3
)

#Onehot
onehot = encoding.OneHotEncoder(variables=best_features, ignore_format=True)


#%%
# MODEL
# model = linear_model.LogisticRegression(
#   penalty=None,
#   random_state=42,
#   max_iter=10000000
# )
# model = naive_bayes.BernoulliNB()
model = ensemble.RandomForestClassifier(
  random_state=42,
  n_jobs=2,
)

params = {
  "min_samples_leaf": [15, 20, 25, 30, 50],
  "n_estimators": [100, 200, 500, 1000],
  "criterion": ['gini', 'entropy', 'log_loss'],
}

grid = model_selection.GridSearchCV(model, params, cv=3, scoring='roc_auc', verbose=4)

model_pipeline = pipeline.Pipeline(
  steps=[
    ('Discretizar', tree_discretization),
    ('Onehot', onehot),
    ('Grid', grid)
  ]
)

with mlflow.start_run(run_name=model.__str__()):
  mlflow.sklearn.autolog()

  grid.fit(X_train[best_features], y_train)

  y_train_predict = grid.predict(X_train[best_features])
  y_train_proba = grid.predict_proba(X_train[best_features])[:,1]

  acc_train = metrics.accuracy_score(y_train, y_train_predict)
  auc_train = metrics.roc_auc_score(y_train, y_train_proba)
  roc_train = metrics.roc_curve(y_train, y_train_proba)
  print(f"Acurácia Treino: {acc_train}")
  print(f"ROC Treino: {auc_train}")

  y_test_predict = grid.predict(X_test[best_features])
  y_test_proba = grid.predict_proba(X_test[best_features])[:,1]

  acc_test = metrics.accuracy_score(y_test, y_test_predict)
  auc_test = metrics.roc_auc_score(y_test, y_test_proba)
  roc_test = metrics.roc_curve(y_test, y_test_proba)
  print(f"Acurácia Test: {acc_test}")
  print(f"ROC Test: {auc_test}")

  y_oot_predict = grid.predict(oot[best_features])
  y_oot_proba = grid.predict_proba(oot[best_features])[:,1]

  acc_oot = metrics.accuracy_score(oot[target], y_oot_predict)
  auc_oot = metrics.roc_auc_score(oot[target], y_oot_proba)
  roc_oot = metrics.roc_curve(oot[target], y_oot_proba)
  print(f"Acurácia oot: {acc_oot}")
  print(f"ROC oot: {auc_oot}")

  mlflow.log_metrics({
    "acc_train": acc_train,
    "auc_train": auc_train,
    "acc_test": acc_test,
    "auc_test": auc_test,
    "acc_oot": acc_oot,
    "auc_oot": auc_oot,
  })

#%%
plt.plot(roc_train[0], roc_train[1])
plt.plot(roc_test[0], roc_test[1])
plt.plot(roc_oot[0], roc_oot[1])
plt.plot([0,1], [0,1], '--', color='black')
plt.grid(True)
plt.ylabel("Sensibilidade")
plt.xlabel("1 - Especificidade")
plt.title("Curva ROC")
plt.legend([
  f"Treino: {100*auc_train:.2f}",
  f"Teste: {100*auc_test:.2f}",
  f"Out-of-Time: {100*auc_oot:.2f}",
])

plt.show()
#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, naive_bayes

df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df

df['aprovado'] = (df['nota'] > 5).astype(int)
df

#%%
reg = linear_model.LogisticRegression(penalty=None, fit_intercept=True)
reg.fit(df[['cerveja']], df['aprovado'])

#%%
reg_predict = reg.predict(df[['cerveja']].drop_duplicates())
reg_predict_proba = reg.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

#%%
plt.figure(dpi=400)
plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.grid(True)
plt.title("Relação Cerveaja x Aprovação")
plt.xlabel("Cervejas")
plt.ylabel("Aprovados")
plt.plot(df['cerveja'].drop_duplicates(), reg_predict)
plt.plot(df['cerveja'].drop_duplicates(), reg_predict_proba)
plt.hlines(0.5, xmin=1, xmax=9, linestyles='--', colors='black')
plt.legend(['Observado', 'Reg Predict', 'Reg Proba'])
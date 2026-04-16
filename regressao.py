#%%
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df.head()

#%%
# Aprendizado de máquina sendo feito aqui!
X = df[['cerveja']]
y = df['nota']

reg = linear_model.LinearRegression()
reg.fit(X, y)

#%%
# Coeficientes
a, b = reg.intercept_, reg.coef_[0]
print(a, b)

#%%
predict = reg.predict(X.drop_duplicates())

#%%
plt.figure(dpi=400)
plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title("Relação Cerveaja x Nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")
plt.plot(X.drop_duplicates()['cerveja'], predict)
plt.legend(['Observado',
            f'y = {a:.3f} + {b:.3f} x',
            ])
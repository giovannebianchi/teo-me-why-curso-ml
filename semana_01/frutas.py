#%%
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_excel('data/dados_frutas.xlsx')
df

#%%
# Criando a árvore
arvore = tree.DecisionTreeClassifier(random_state=42)

#%% 
# ML em si
y = df['Fruta']
caracteristicas = ['Arredondada', 'Suculenta', 'Vermelha', 'Doce']
X = df[caracteristicas]
 
#%%
arvore.fit(X=X, y=y)

#%%
arvore.predict([[0,1,1,1]])

#%%
plt.figure(dpi=400)
tree.plot_tree(arvore, feature_names=caracteristicas, class_names=arvore.classes_,filled=True)

#%%
proba = arvore.predict_proba([[1,1,1,1]])[0]
pd.Series(proba, index=arvore.classes_)
#%%
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_parquet("data/dados_clones.parquet")
df.head()

#%%
features = ['Massa(em kilos)', 'Estatura(cm)']
target = 'Status '

X = df[features]
y = df[target]

X = X.replace({
  "Aayla Secura": 1, "Mace Windu": 2,
  "Obi-Wan Kenobi": 3, "Shaak Ti": 4,
  "Yoda": 5
})
# %%
model = tree.DecisionTreeClassifier(random_state=42)
model.fit(X=X, y=y)

#%%
plt.figure(dpi=400)
tree.plot_tree(model, feature_names=features,
               class_names=model.classes_,
               filled=True, max_depth=3)
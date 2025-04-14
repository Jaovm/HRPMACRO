import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

# Simular preços de 4 ativos com correlação conhecida
np.random.seed(42)
n = 250
dates = pd.date_range(end=pd.Timestamp.today(), periods=n)

# Simulação: ativos A e B altamente correlacionados; C e D menos
ret_A = np.random.normal(0.001, 0.02, size=n)
ret_B = ret_A + np.random.normal(0, 0.005, size=n)  # correlacionado com A
ret_C = np.random.normal(0.001, 0.02, size=n)
ret_D = ret_C + np.random.normal(0, 0.005, size=n)  # correlacionado com C

df = pd.DataFrame({
    'A': (1 + ret_A).cumprod(),
    'B': (1 + ret_B).cumprod(),
    'C': (1 + ret_C).cumprod(),
    'D': (1 + ret_D).cumprod(),
}, index=dates)

# Calcular retornos
retornos = df.pct_change().dropna()

# 1. Mostrar matriz de correlação
plt.figure(figsize=(6, 4))
sns.heatmap(retornos.corr(), annot=True, cmap='coolwarm')
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.show()

# 2. Matriz de distância
dist = np.sqrt(((1 - retornos.corr()) / 2).fillna(0))
linkage_matrix = linkage(squareform(dist), method='ward')

# 3. Dendrograma
plt.figure(figsize=(6, 3))
dendrogram(linkage_matrix, labels=retornos.columns)
plt.title("Dendrograma (Clustering Hierárquico)")
plt.tight_layout()
plt.show()

# 4. HRP - Quasi Diagonal
def get_quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0]*2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i+1)
        sort_ix = pd.concat([sort_ix, df1])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()

sort_ix = get_quasi_diag(linkage_matrix)
sorted_tickers = [retornos.columns[i] for i in sort_ix]

# 5. HRP - Recursive Bisection
cov = LedoitWolf().fit(retornos).covariance_

def get_cluster_var(cov, cluster_items):
    cov_slice = cov[np.ix_(cluster_items, cluster_items)]
    w_ = 1. / np.diag(cov_slice)
    w_ /= w_.sum()
    return np.dot(w_, np.dot(cov_slice, w_))

def recursive_bisection(cov, sort_ix):
    w = pd.Series(1, index=sort_ix)
    cluster_items = [sort_ix]
    while len(cluster_items) > 0:
        cluster_items = [i[j:k] for i in cluster_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for i in range(0, len(cluster_items), 2):
            c0 = cluster_items[i]
            c1 = cluster_items[i + 1]
            var0 = get_cluster_var(cov, c0)
            var1 = get_cluster_var(cov, c1)
            alpha = 1 - var0 / (var0 + var1)
            w[c0] *= alpha
            w[c1] *= 1 - alpha
    return w

weights = recursive_bisection(cov, list(range(len(sorted_tickers))))
pesos_hrp = pd.Series(weights.values, index=sorted_tickers)
print("Pesos HRP:")
print(pesos_hrp.round(4))

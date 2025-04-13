
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# ============================ DICIONÁRIOS DE SETORES ============================
setores_por_ticker = {
    'AGRO3.SA': 'Agro',
    'BBAS3.SA': 'Banco',
    'BBSE3.SA': 'Seguradora',
    'BPAC11.SA': 'Banco',
    'EGIE3.SA': 'Energia',
    'ITUB3.SA': 'Banco',
    'PRIO3.SA': 'Petróleo',
    'PSSA3.SA': 'Seguradora',
    'SAPR3.SA': 'Saneamento',
    'SBSP3.SA': 'Saneamento',
    'VIVT3.SA': 'Telecomunicação',
    'WEGE3.SA': 'Indústria',
    'TOTS3.SA': 'Tecnologia',
    'B3SA3.SA': 'Bolsa',
    'TAEE3.SA': 'Energia',
}

setores_por_cenario = {
    "Expansionista": ['Tecnologia', 'Indústria', 'Agro', 'Consumo discricionário'],
    "Neutro": ['Banco', 'Seguradora', 'Telecomunicação', 'Saneamento'],
    "Restritivo": ['Energia', 'Petróleo', 'Saneamento', 'Consumo básico']
}

# ============================ FUNÇÕES DE MACRO ============================
def get_bcb(code):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados/ultimos/1?formato=json"
    r = requests.get(url)
    return float(r.json()[0]['valor'].replace(",", ".")) if r.status_code == 200 else None

def obter_macro():
    return {
        "selic": get_bcb(432),
        "ipca": get_bcb(433),
        "meta_inflacao": 3.0
    }

def classificar_cenario_macro(m):
    if m['ipca'] > m['meta_inflacao'] and m['selic'] >= 12:
        return "Restritivo"
    elif m['ipca'] < m['meta_inflacao'] and m['selic'] < 10:
        return "Expansionista"
    else:
        return "Neutro"

# ============================ HRP ============================
def get_returns(tickers):
    dados = yf.download(tickers, period="3y", auto_adjust=True)['Close']
    return dados.pct_change().dropna()

def correlacao_dist(corr):
    return ((1 - corr) / 2) ** 0.5

def get_hrp_weights(retornos, setores, setores_favorecidos):
    corr = retornos.corr()
    dist = correlacao_dist(corr)
    link = linkage(squareform(dist), method='single')
    dendro = dendrogram(link, no_plot=True)
    ordenados = dendro['leaves']
    tickers = list(retornos.columns)
    tickers_ordenados = [tickers[i] for i in ordenados]

    def get_cluster_var(cov, items):
        sub_cov = cov.loc[items, items]
        w = np.linalg.inv(sub_cov).sum(axis=1)
        w /= w.sum()
        return np.dot(np.dot(w, sub_cov), w.T)

    def recursive_bisect(cov, tickers_ordenados):
        w = pd.Series(1, index=tickers_ordenados)
        clusters = [tickers_ordenados]
        while len(clusters) > 0:
            cluster = clusters.pop(0)
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            cluster1 = cluster[:split]
            cluster2 = cluster[split:]
            var1 = get_cluster_var(cov, cluster1)
            var2 = get_cluster_var(cov, cluster2)
            alpha = 1 - var1 / (var1 + var2)
            w[cluster1] *= alpha
            w[cluster2] *= 1 - alpha
            clusters += [cluster1, cluster2]
        return w

    cov = LedoitWolf().fit(retornos).covariance_
    cov_df = pd.DataFrame(cov, index=retornos.columns, columns=retornos.columns)
    pesos = recursive_bisect(cov_df, tickers_ordenados)

    # Ajustar pesos com base no cenário macro
    pesos *= [1.2 if setores[t] in setores_favorecidos else 1 for t in pesos.index]
    pesos /= pesos.sum()
    return pesos

# ============================ STREAMLIT ============================
st.set_page_config("Carteira com HRP + Macro", layout="wide")
st.title("📊 Carteira com Alocação Hierárquica Ponderada + Macroeconomia")

# Macro
macro = obter_macro()
cenario = classificar_cenario_macro(macro)
st.sidebar.metric("Selic (%)", f"{macro['selic']:.2f}")
st.sidebar.metric("Inflação (%)", f"{macro['ipca']:.2f}")
st.sidebar.metric("Meta Inflação (%)", f"{macro['meta_inflacao']:.2f}")
st.sidebar.info(f"**Cenário Atual:** {cenario}")

# Carteira atual (fixa)
pesos_atuais = {
    'AGRO3.SA': 0.10, 'BBAS3.SA': 0.012, 'BBSE3.SA': 0.065, 'BPAC11.SA': 0.106,
    'EGIE3.SA': 0.05, 'ITUB3.SA': 0.005, 'PRIO3.SA': 0.15, 'PSSA3.SA': 0.15,
    'SAPR3.SA': 0.067, 'SBSP3.SA': 0.04, 'VIVT3.SA': 0.064, 'WEGE3.SA': 0.15,
    'TOTS3.SA': 0.01, 'B3SA3.SA': 0.001, 'TAEE3.SA': 0.03
}

tickers = list(pesos_atuais.keys())
retornos = get_returns(tickers)
pesos_hrp = get_hrp_weights(retornos, setores_por_ticker, setores_por_cenario[cenario])

# Mostrar comparação
df_pesos = pd.DataFrame({
    "Setor": [setores_por_ticker[t] for t in tickers],
    "Peso Atual (%)": [pesos_atuais[t] * 100 for t in tickers],
    "Peso Sugerido HRP (%)": (pesos_hrp[tickers] * 100).round(2)
}, index=tickers)

st.subheader("📌 Comparação de Pesos na Carteira")
st.dataframe(df_pesos.sort_values("Peso Sugerido HRP (%)", ascending=False))

# Aporte mensal
st.subheader("💰 Sugestão de Aporte Mensal")
aporte = st.number_input("Valor do aporte (R$)", min_value=100.0, value=1000.0, step=100.0)

nova_carteira = {t: pesos_atuais[t] + (pesos_hrp[t] * aporte / 100000) for t in tickers}
df_aporte = pd.DataFrame({
    "Setor": [setores_por_ticker[t] for t in tickers],
    "Novo Peso (%)": [nova_carteira[t] * 100 for t in tickers]
}, index=tickers)
st.dataframe(df_aporte.sort_values("Novo Peso (%)", ascending=False))

st.success("✅ Aporte distribuído com base na alocação HRP favorecendo setores do cenário atual.")

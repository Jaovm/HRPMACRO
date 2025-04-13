import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.optimize import minimize

# ========= DICIONÁRIOS ==========

setores_por_ticker = {
    'WEGE3.SA': 'Indústria', 'PETR4.SA': 'Energia', 'VIVT3.SA': 'Utilidades',
    'EGIE3.SA': 'Energia', 'ITUB4.SA': 'Financeiro', 'LREN3.SA': 'Consumo discricionário',
    'ABEV3.SA': 'Consumo básico', 'B3SA3.SA': 'Financeiro', 'MGLU3.SA': 'Consumo discricionário',
    'HAPV3.SA': 'Saúde', 'RADL3.SA': 'Saúde', 'RENT3.SA': 'Consumo discricionário',
    'VALE3.SA': 'Indústria', 'TOTS3.SA': 'Tecnologia', 'AGRO3.SA': 'Agronegócio',
    'BBAS3.SA': 'Financeiro', 'BBSE3.SA': 'Seguradoras', 'BPAC11.SA': 'Financeiro',
    'PRIO3.SA': 'Petróleo', 'PSSA3.SA': 'Seguradoras', 'SAPR3.SA': 'Utilidades',
    'SBSP3.SA': 'Utilidades', 'TAEE3.SA': 'Energia'
}

setores_por_cenario = {
    "Expansionista": ['Consumo discricionário', 'Tecnologia', 'Indústria', 'Agronegócio'],
    "Neutro": ['Saúde', 'Financeiro', 'Utilidades', 'Varejo', 'Seguradoras'],
    "Restritivo": ['Utilidades', 'Energia', 'Saúde', 'Consumo básico', 'Petróleo']
}

empresas_exportadoras = ['AGRO3.SA', 'PRIO3.SA']

# ========= MACRO ==========
def get_bcb(code):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados/ultimos/1?formato=json"
    r = requests.get(url)
    return float(r.json()[0]['valor'].replace(",", ".")) if r.status_code == 200 else None

def obter_preco_petroleo():
    try:
        dados = yf.Ticker("CL=F").history(period="5d")
        if not dados.empty and 'Close' in dados.columns:
            return float(dados['Close'].dropna().iloc[-1])
        else:
            return None
    except Exception as e:
        st.error(f"Erro ao obter preço do petróleo: {e}")
        return None

def obter_macro():
    return {
        "selic": get_bcb(432),
        "ipca": get_bcb(433),
        "dolar": get_bcb(1),
        "petroleo": obter_preco_petroleo()
    }

def classificar_cenario_macro(m):
    if m['ipca'] > 5 or m['selic'] > 12:
        return "Restritivo"
    elif m['ipca'] < 4 and m['selic'] < 10:
        return "Expansionista"
    else:
        return "Neutro"

# ========= PREÇO ALVO ==========

def obter_preco_diario_ajustado(tickers, periodo='7y'):
    dados = yf.download(tickers, period=periodo, group_by='ticker', auto_adjust=True)
    if len(tickers) == 1:
        return dados['Adj Close'].to_frame()
    else:
        return dados['Adj Close']


def otimizar_carteira_sharpe(tickers):
    dados = obter_preco_diario_ajustado(tickers)
    retornos = dados.pct_change().dropna()
    media_retorno = retornos.mean()
    cov = LedoitWolf().fit(retornos).covariance_

    def sharpe_neg(pesos):
        retorno_esperado = np.dot(pesos, media_retorno)
        volatilidade = np.sqrt(np.dot(pesos.T, np.dot(cov, pesos)))
        return -retorno_esperado / volatilidade if volatilidade != 0 else 0

    n = len(tickers)
    pesos_iniciais = np.array([1/n] * n)
    limites = [(0, 1) for _ in range(n)]
    restricoes = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    resultado = minimize(sharpe_neg, pesos_iniciais, method='SLSQP', bounds=limites, constraints=restricoes)

    if resultado.success:
        return resultado.x
    else:
        return None

def obter_preco_alvo(ticker):
    try:
        return yf.Ticker(ticker).info.get('targetMeanPrice', None)
    except:
        return None

def obter_preco_atual(ticker):
    try:
        return yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    except:
        return None

# ========= FILTRAR AÇÕES ==========
def calcular_score(preco_atual, preco_alvo, favorecido, ticker, macro):
    upside = (preco_alvo - preco_atual) / preco_atual
    bonus = 0.1 if favorecido else 0
    if ticker in empresas_exportadoras:
        if macro['dolar'] and macro['dolar'] > 5:
            bonus += 0.05
        if macro['petroleo'] and macro['petroleo'] > 80:
            bonus += 0.05
    return upside + bonus

def filtrar_ativos_validos(carteira, cenario, macro):
    setores_bons = setores_por_cenario[cenario]
    ativos_validos = []

    for ticker in carteira:
        setor = setores_por_ticker.get(ticker, None)
        preco_atual = obter_preco_atual(ticker)
        preco_alvo = obter_preco_alvo(ticker)

        if preco_atual is None or preco_alvo is None:
            continue
        if preco_atual < preco_alvo:
            favorecido = setor in setores_bons
            score = calcular_score(preco_atual, preco_alvo, favorecido, ticker, macro)
            ativos_validos.append({
                "ticker": ticker,
                "setor": setor,
                "preco_atual": preco_atual,
                "preco_alvo": preco_alvo,
                "favorecido": favorecido,
                "score": score
            })

    ativos_validos.sort(key=lambda x: x['score'], reverse=True)
    return ativos_validos

# ========= OTIMIZAÇÃO ==========
def obter_preco_diario_ajustado(tickers):
    dados_brutos = yf.download(tickers, period="3y", auto_adjust=False)

    if isinstance(dados_brutos.columns, pd.MultiIndex):
        if 'Adj Close' in dados_brutos.columns.get_level_values(0):
            return dados_brutos['Adj Close']
        elif 'Close' in dados_brutos.columns.get_level_values(0):
            return dados_brutos['Close']
        else:
            raise ValueError("Colunas 'Adj Close' ou 'Close' não encontradas nos dados.")
    else:
        if 'Adj Close' in dados_brutos.columns:
            return dados_brutos[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
        elif 'Close' in dados_brutos.columns:
            return dados_brutos[['Close']].rename(columns={'Close': tickers[0]})
        else:
            raise ValueError("Coluna 'Adj Close' ou 'Close' não encontrada nos dados.")

def otimizar_carteira_hrp(tickers):
    dados = obter_preco_diario_ajustado(tickers)
    retornos = dados.pct_change().dropna()
    dist = np.sqrt(((1 - retornos.corr()) / 2).fillna(0))
    linkage_matrix = linkage(squareform(dist), method='single')

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
    cov = LedoitWolf().fit(retornos).covariance_
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()

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

    hrp_weights = recursive_bisection(cov, list(range(len(tickers))))
    return hrp_weights.values

# ========= STREAMLIT ==========
st.set_page_config(page_title="Sugestão de Carteira", layout="wide")
st.title("📊 Sugestão e Otimização de Carteira com Base no Cenário Macroeconômico")

macro = obter_macro()
cenario = classificar_cenario_macro(macro)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cenário Macroeconômico Atual")
    st.write(f"Cenário: **{cenario}**")
    st.write(f"IPCA: {macro['ipca']}%")
    st.write(f"Taxa SELIC: {macro['selic']}%")
    st.write(f"Preço do Petróleo: US${macro['petroleo']}")

with col2:
    st.subheader("Carteira de Ativos")
    st.write("Lista de ações e seus respectivos setores:")
    st.write(setores_por_ticker)

ativos_validos = filtrar_ativos_validos(setores_por_ticker.keys(), cenario, macro)
st.subheader("Ações Mais Indicadas para o Cenário Atual")
for ativo in ativos_validos:
    st.write(f"**{ativo['ticker']}** - Setor: {ativo['setor']} - Score: {ativo['score']}")

# Executar a otimização da carteira com Sharpe
optimal_weights = otimizar_carteira_sharpe([ativo['ticker'] for ativo in ativos_validos])
st.write("Pesos Otimizados (Sharpe):", optimal_weights)

# Executar a otimização HRP
hrp_weights = otimizar_carteira_hrp([ativo['ticker'] for ativo in ativos_validos])
st.write("Pesos Otimizados (HRP):", hrp_weights)

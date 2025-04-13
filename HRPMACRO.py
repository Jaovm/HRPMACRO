import streamlit as st
import requests
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import yfinance as yf

# Funções auxiliares para HRP
def correl_to_dist(corr):
    return np.sqrt(0.5 * (1 - corr))

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

def get_cluster_var(cov, items):
    cov_ = cov.loc[items, items]
    w_ = np.linalg.inv(cov_).sum(axis=1)
    w_ /= w_.sum()
    return np.dot(w_, np.dot(cov_, w_))

def get_recursive_bisection(cov, sort_ix):
    w = pd.Series(1, index=sort_ix)
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [i[j:k] for i in c_items for j, k in ((0, len(i)//2), (len(i)//2, len(i))) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]
            c_items1 = c_items[i + 1]
            c_var0 = get_cluster_var(cov, c_items0)
            c_var1 = get_cluster_var(cov, c_items1)
            alpha = 1 - c_var0 / (c_var0 + c_var1)
            w[c_items0] *= alpha
            w[c_items1] *= 1 - alpha
    return w

# Função para obter dados econômicos da API do Banco Central (SGS)
def get_selic():
    url = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.4189/dados?formato=csv'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.text.splitlines()
        if len(data) > 1:
            selic_data = [line.split(';') for line in data]
            try:
                # Remover aspas da string e converter para float
                selic_value = selic_data[-1][1].replace('"', '').replace(',', '.')
                return float(selic_value)  # Última taxa Selic
            except (IndexError, ValueError) as e:
                st.error(f"Erro ao acessar os dados da taxa Selic: {e}")
                return None
        else:
            st.error("Nenhum dado encontrado na resposta da API da Selic.")
            return None
    else:
        st.error(f"Erro ao acessar a API do Banco Central. Código de status: {response.status_code}")
        return None

def get_inflacao():
    url = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=csv'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.text.splitlines()
        if len(data) > 1:
            inflacao_data = [line.split(';') for line in data]
            try:
                return float(inflacao_data[-1][1].replace(',', '.'))  # Última inflação
            except (IndexError, ValueError) as e:
                st.error(f"Erro ao acessar os dados de inflação: {e}")
                return None
        else:
            st.error("Nenhum dado encontrado na resposta da API da inflação.")
            return None
    else:
        st.error(f"Erro ao acessar a API do Banco Central. Código de status: {response.status_code}")
        return None

# Dados da carteira e setores
pesos_atuais = {
    'AGRO3.SA': 0.10,
    'BBAS3.SA': 0.012,
    'BBSE3.SA': 0.065,
    'BPAC11.SA': 0.106,
    'EGIE3.SA': 0.05,
    'ITUB3.SA': 0.005,
    'PRIO3.SA': 0.15,
    'PSSA3.SA': 0.15,
    'SAPR3.SA': 0.067,
    'SBSP3.SA': 0.04,
    'VIVT3.SA': 0.064,
    'WEGE3.SA': 0.15,
    'TOTS3.SA': 0.01,
    'B3SA3.SA': 0.001,
    'TAEE3.SA': 0.03
}

setores_por_ticker = {
    'AGRO3.SA': 'Consumo básico',
    'BBAS3.SA': 'Financeiro',
    'BBSE3.SA': 'Financeiro',
    'BPAC11.SA': 'Financeiro',
    'EGIE3.SA': 'Utilidades',
    'ITUB3.SA': 'Financeiro',
    'PRIO3.SA': 'Energia',
    'PSSA3.SA': 'Financeiro',
    'SAPR3.SA': 'Utilidades',
    'SBSP3.SA': 'Utilidades',
    'VIVT3.SA': 'Comunicações',
    'WEGE3.SA': 'Indústria',
    'TOTS3.SA': 'Tecnologia',
    'B3SA3.SA': 'Financeiro',
    'TAEE3.SA': 'Utilidades'
}

# Determinar automaticamente o cenário macroeconômico
inflacao_anual = get_inflacao()  # Obter inflação
selic = get_selic()  # Obter taxa Selic
meta_inflacao = 3.0  # Meta de inflação do Banco Central

if inflacao_anual and inflacao_anual > meta_inflacao and selic >= 14.25:
    cenario = 'Restritivo'
    setores_favorecidos = ['Utilidades', 'Energia', 'Consumo básico', 'Saúde']
elif inflacao_anual and inflacao_anual <= meta_inflacao and selic <= 10.0:
    cenario = 'Expansivo'
    setores_favorecidos = ['Tecnologia', 'Consumo discricionário', 'Financeiro']
else:
    cenario = 'Neutro'
    setores_favorecidos = ['Indústria', 'Comunicações', 'Financeiro']

# App
st.title("Sugestão de Alocação de Aporte com HRP")

st.write(f"**Cenário Macroeconômico Atual:** {cenario}")
st.write(f"**Setores Favorecidos:** {', '.join(setores_favorecidos)}")

aporte = st.number_input("Valor do novo aporte (R$)", min_value=100.0, value=5000.0, step=100.0)

if st.button("Gerar sugestão de alocação"):
    ativos_fav = [t for t in pesos_atuais.keys() if setores_por_ticker.get(t) in setores_favorecidos]
    if not ativos_fav:
        st.warning("Nenhum ativo da carteira pertence aos setores favorecidos no cenário atual.")
    else:
        # Ajustar alocação usando HRP com base nos ativos favorecidos
        st.write("Ativos favorecidos:", ativos_fav)
        # Aqui você pode aplicar o método de HRP para gerar a alocação do novo aporte com base nos ativos favorecidos.
        st.write("Nova alocação sugerida com HRP")

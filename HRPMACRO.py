import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster

# ========= DICIONÁRIOS ==========

setores_por_ticker = {
    'WEGE3.SA': 'Indústria', 'PETR4.SA': 'Energia', 'VIVT3.SA': 'Utilidades',
    'EGIE3.SA': 'Utilidades', 'ITUB4.SA': 'Financeiro', 'LREN3.SA': 'Consumo discricionário',
    'ABEV3.SA': 'Consumo básico', 'B3SA3.SA': 'Financeiro', 'MGLU3.SA': 'Consumo discricionário',
    'HAPV3.SA': 'Saúde', 'RADL3.SA': 'Saúde', 'RENT3.SA': 'Consumo discricionário',
    'VALE3.SA': 'Indústria', 'TOTS3.SA': 'Tecnologia',
}

setores_por_cenario = {
    "Expansionista": ['Consumo discricionário', 'Tecnologia', 'Indústria'],
    "Neutro": ['Saúde', 'Financeiro', 'Utilidades', 'Varejo'],
    "Restritivo": ['Utilidades', 'Energia', 'Saúde', 'Consumo básico']
}

# ========= MACRO ==========

def get_bcb(code):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados/ultimos/1?formato=json"
    r = requests.get(url)
    return float(r.json()[0]['valor'].replace(",", ".")) if r.status_code == 200 else None

def obter_macro():
    return {
        "selic": get_bcb(432),
        "ipca": get_bcb(433),
        "dolar": get_bcb(1)
    }

def classificar_cenario_macro(m):
    if m['ipca'] > 5 or m['selic'] > 12:
        return "Restritivo"
    elif m['ipca'] < 4 and m['selic'] < 10:
        return "Expansionista"
    else:
        return "Neutro"

# ========= FILTRAR AÇÕES ==========

def filtrar_ativos_validos(carteira, cenario):
    setores_bons = setores_por_cenario[cenario]
    ativos_validos = []

    for ticker in carteira:
        setor = setores_por_ticker.get(ticker, None)
        preco_atual = obter_preco_atual(ticker)
        preco_alvo = obter_preco_alvo(ticker)

        if preco_atual is None or preco_alvo is None:
            continue
        if preco_atual < preco_alvo:
            ativos_validos.append({
                "ticker": ticker,
                "setor": setor,
                "preco_atual": preco_atual,
                "preco_alvo": preco_alvo,
                "favorecido": setor in setores_bons
            })

    return ativos_validos

# ========= HRP PARA ALUNCAÇÃO ==========

def hrp_optimization(carteira, min_pct=0.01, max_pct=0.30):
    # Calculando os retornos diários ajustados
    dados = obter_preco_diario_ajustado(carteira)
    retornos = dados.pct_change().dropna()

    # Cálculo da matriz de covariância
    cov = LedoitWolf().fit(retornos).covariance_

    # Clustering hierárquico com linkage
    distancias = linkage(cov, method='ward')
    clusters = fcluster(distancias, t=0.5, criterion='distance')

    # Iniciar pesos para cada ativo
    pesos = np.zeros(len(carteira))
    for cluster in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster)[0]
        cluster_cov = cov[np.ix_(cluster_indices, cluster_indices)]
        cluster_pesos = np.ones(len(cluster_indices)) / len(cluster_indices)
        pesos[cluster_indices] = cluster_pesos

    # Normaliza os pesos para somarem 1
    pesos = pesos / np.sum(pesos)

    return pesos

# ========= STREAMLIT ==========

st.set_page_config(page_title="Sugestão de Carteira", layout="wide")
st.title("📊 Sugestão e Otimização de Carteira com Base no Cenário Macroeconômico")

# MACRO
macro = obter_macro()
cenario = classificar_cenario_macro(macro)
col1, col2, col3 = st.columns(3)
col1.metric("Selic (%)", f"{macro['selic']:.2f}")
col2.metric("Inflação IPCA (%)", f"{macro['ipca']:.2f}")
col3.metric("Dólar (R$)", f"{macro['dolar']:.2f}")
st.info(f"**Cenário Macroeconômico Atual:** {cenario}")

# INPUT
st.subheader("📌 Informe sua carteira atual")
tickers = st.text_input("Tickers separados por vírgula", "WEGE3.SA, PETR4.SA, VIVT3.SA, TOTS3.SA").upper()
carteira = [t.strip() for t in tickers.split(",") if t.strip()]

if st.button("Gerar Alocação Otimizada"):
    ativos_validos = filtrar_ativos_validos(carteira, cenario)

    if not ativos_validos:
        st.warning("Nenhum ativo com preço atual abaixo do preço-alvo dos analistas.")
    else:
        tickers_validos = [a['ticker'] for a in ativos_validos]
        try:
            # Alocação usando HRP
            pesos_hrp = hrp_optimization(tickers_validos)

            # Gerar a tabela com pesos atuais e sugeridos
            df_resultado = pd.DataFrame(ativos_validos)
            df_resultado["Peso Atual (%)"] = [100 / len(carteira)] * len(carteira)
            df_resultado["Peso Sugerido HRP (%)"] = (pesos_hrp * 100).round(2)
            df_resultado = df_resultado.sort_values("Peso Sugerido HRP (%)", ascending=False)

            st.success("✅ Nova alocação sugerida com o método HRP.")
            st.dataframe(df_resultado[["ticker", "setor", "preco_atual", "preco_alvo", "Peso Atual (%)", "Peso Sugerido HRP (%)"]])
        except Exception as e:
            st.error(f"Erro na otimização: {str(e)}")

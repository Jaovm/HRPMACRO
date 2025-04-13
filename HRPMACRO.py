import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime

# ========================
# Funções auxiliares
# ========================

def obter_indicadores_macro_bcb():
    indicadores = {}

    # IPCA - Índice de Preços ao Consumidor Amplo
    ipca_url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados/ultimos/1?formato=json"
    ipca_response = requests.get(ipca_url)
    if ipca_response.status_code == 200:
        ipca_data = ipca_response.json()
        indicadores["Inflação IPCA (12m)"] = f"{ipca_data[0]['valor']}%"

    # Selic - Taxa básica de juros
    selic_url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json"
    selic_response = requests.get(selic_url)
    if selic_response.status_code == 200:
        selic_data = selic_response.json()
        indicadores["Taxa Selic"] = f"{selic_data[0]['valor']}%"

    # Dólar - Taxa de câmbio
    dolar_url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados/ultimos/1?formato=json"
    dolar_response = requests.get(dolar_url)
    if dolar_response.status_code == 200:
        dolar_data = dolar_response.json()
        indicadores["Dólar (R$)"] = f"{dolar_data[0]['valor']}"

    return indicadores

def obter_preco_acao(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1d")
        return df['Close'].iloc[-1]
    except Exception:
        return None

def gerar_sugestoes(carteira, precos_teto, macro_estavel=True):
    sugestoes = []
    for ticker, preco_teto in precos_teto.items():
        preco_atual = obter_preco_acao(ticker)
        if preco_atual is None:
            continue
        if preco_atual < preco_teto and macro_estavel:
            sugestoes.append({
                "Ticker": ticker,
                "Preço Atual": round(preco_atual, 2),
                "Preço Teto": preco_teto,
                "Sugestão": "Comprar"
            })
    return pd.DataFrame(sugestoes)

# ========================
# App Streamlit
# ========================

st.title("📊 Análise Macroeconômica com Dados Reais + Sugestões de Compra")

st.subheader("1. Cenário Macroeconômico Atual")
indicadores = obter_indicadores_macro_bcb()
for nome, valor in indicadores.items():
    st.markdown(f"- **{nome}**: {valor}")

macro_estavel = st.checkbox("Considerar cenário macroeconômico estável para sugestões de compra?", value=True)

st.subheader("2. Sua Carteira de Investimentos")
carteira_input = st.text_area("Informe os tickers separados por vírgula (ex: WEGE3.SA,EGIE3.SA):")
carteira = [ticker.strip().upper() for ticker in carteira_input.split(",") if ticker.strip()]

st.subheader("3. Preços Teto dos Ativos")
precos_teto = {}
for ticker in carteira:
    preco = st.number_input(f"Preço teto para {ticker}:", min_value=0.0, step=0.01)
    precos_teto[ticker] = preco

if st.button("Gerar Sugestões de Compra"):
    with st.spinner("Analisando preços..."):
        sugestoes_df = gerar_sugestoes(carteira, precos_teto, macro_estavel)
    if sugestoes_df.empty:
        st.warning("Nenhuma sugestão de compra com base nos critérios atuais.")
    else:
        st.success("Sugestões de compra geradas com base no cenário atual e nos preços teto.")
        st.dataframe(sugestoes_df)

import streamlit as st
import pandas as pd
import yfinance as yf
import requests

# ========== FUNÇÕES AUXILIARES ==========

def obter_dados_macro_bcb():
    def get_bcb_data(code):
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados/ultimos/1?formato=json"
        response = requests.get(url)
        if response.status_code == 200:
            return float(response.json()[0]['valor'].replace(',', '.'))
        return None

    return {
        "selic": get_bcb_data(432),
        "ipca": get_bcb_data(433),
        "dolar": get_bcb_data(1)
    }

def classificar_cenario_macro(dados):
    if dados['ipca'] > 5 or dados['selic'] > 12:
        return "Restritivo"
    elif dados['ipca'] < 4 and dados['selic'] < 10:
        return "Expansionista"
    else:
        return "Neutro"

def obter_preco_atual(ticker):
    try:
        return yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    except:
        return None

def obter_preco_alvo_yf(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get('targetMeanPrice', None)
    except:
        return None

def gerar_sugestoes(carteira):
    sugestoes = []
    for ticker in carteira:
        preco_atual = obter_preco_atual(ticker)
        preco_alvo = obter_preco_alvo_yf(ticker)

        if preco_atual is None or preco_alvo is None:
            continue

        if preco_atual < preco_alvo:
            sugestoes.append({
                "Ticker": ticker,
                "Preço Atual (R$)": round(preco_atual, 2),
                "Preço Alvo Médio (R$)": round(preco_alvo, 2),
                "Sugestão": "Comprar"
            })

    return pd.DataFrame(sugestoes)

# ========== INTERFACE STREAMLIT ==========

st.title("📈 Análise Macroeconômica + Sugestões de Compra com Preço-Alvo dos Analistas")

st.subheader("1. Cenário Macroeconômico Atual")
macro = obter_dados_macro_bcb()
cenario = classificar_cenario_macro(macro)

col1, col2, col3 = st.columns(3)
col1.metric("Selic (%)", f"{macro['selic']:.2f}")
col2.metric("Inflação IPCA (%)", f"{macro['ipca']:.2f}")
col3.metric("Dólar (R$)", f"{macro['dolar']:.2f}")
st.info(f"**Classificação do cenário macroeconômico:** {cenario}")

st.subheader("2. Informe sua Carteira")
tickers_input = st.text_input("Tickers separados por vírgula (ex: WEGE3.SA, PETR4.SA)")
carteira = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if st.button("Gerar Sugestões de Compra"):
    with st.spinner("Buscando dados..."):
        df_sugestoes = gerar_sugestoes(carteira)
    if df_sugestoes.empty:
        st.warning("Nenhuma sugestão gerada com os critérios atuais.")
    else:
        st.success("Sugestões geradas com base no preço-alvo médio dos analistas.")
        st.dataframe(df_sugestoes)

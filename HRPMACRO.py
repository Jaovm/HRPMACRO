import streamlit as st
import yfinance as yf
import pandas as pd
import requests

# --- Função para simular análise macroeconômica ---
def analisar_cenario_macroeconomico():
    # Aqui você pode integrar com APIs como TradingEconomics, FRED, etc.
    # Por enquanto, vamos simular com base em dados fixos
    dados_macro = {
        "Selic": 10.75,
        "Inflação (IPCA)": 4.1,
        "PIB": 2.2,
        "Emprego": "Estável",
        "Cenário geral": "Neutro-positivo"
    }
    return dados_macro

def avaliar_cenario(dados_macro):
    if dados_macro["Selic"] < 11 and dados_macro["Inflação (IPCA)"] < 5:
        return "positivo"
    elif dados_macro["Selic"] > 13 or dados_macro["Inflação (IPCA)"] > 6:
        return "negativo"
    else:
        return "neutro"

# --- Função para obter preço atual do ativo ---
def obter_preco_atual(ticker):
    try:
        dados = yf.Ticker(ticker).history(period="1d")
        preco = dados["Close"].iloc[-1]
        return preco
    except:
        return None

# --- Título ---
st.title("📈 Análise Macro + Sugestões de Compra")

# --- Entrada do usuário ---
st.subheader("Carteira de Ativos")
with st.expander("📋 Insira sua carteira e preço teto por ativo"):
    carteira = st.text_area("Tickers (um por linha, formato: TICKER,PREÇO_TETO)", 
                            "ITUB3.SA,32\nWEGE3.SA,40\nPRIO3.SA,48")
    carteira_dict = {}
    for linha in carteira.strip().split("\n"):
        try:
            ticker, teto = linha.split(",")
            carteira_dict[ticker.strip().upper()] = float(teto)
        except:
            st.warning(f"Linha inválida: {linha}")

# --- Análise macroeconômica ---
st.subheader("🌍 Cenário Macroeconômico")
dados_macro = analisar_cenario_macroeconomico()
st.write(dados_macro)
cenário = avaliar_cenario(dados_macro)
st.markdown(f"**Cenário identificado: `{cenário.upper()}`**")

# --- Sugestões de compra ---
st.subheader("💡 Sugestões de Compra")
if cenário == "negativo":
    st.warning("O cenário macroeconômico atual não é favorável. Sugestão: aguardar.")
else:
    sugestoes = []
    for ticker, preco_teto in carteira_dict.items():
        preco_atual = obter_preco_atual(ticker)
        if preco_atual is None:
            st.error(f"Erro ao obter preço de {ticker}")
            continue
        if preco_atual < preco_teto:
            sugestoes.append((ticker, preco_atual, preco_teto))
    
    if sugestoes:
        df_sugestoes = pd.DataFrame(sugestoes, columns=["Ticker", "Preço Atual", "Preço Teto"])
        st.success("Ativos com preço abaixo do teto:")
        st.dataframe(df_sugestoes)
    else:
        st.info("Nenhum ativo está abaixo do preço teto no momento.")

# --- Rodapé ---
st.markdown("---")
st.caption("Desenvolvido com ❤️ para investidores de longo prazo.")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

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

# ========= PREÇO ALVO ==========
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

# ========= OTIMIZAÇÃO CORRIGIDA ==========
def obter_preco_diario_ajustado(tickers):
    dados_brutos = yf.download(tickers, period="3y", auto_adjust=False)

    if isinstance(dados_brutos.columns, pd.MultiIndex):
        # Vários ativos — usa MultiIndex com 'Adj Close'
        if 'Adj Close' in dados_brutos.columns.get_level_values(0):
            return dados_brutos['Adj Close']
        elif 'Close' in dados_brutos.columns.get_level_values(0):
            return dados_brutos['Close']
        else:
            raise ValueError("Colunas 'Adj Close' ou 'Close' não encontradas nos dados.")
    else:
        # Apenas 1 ativo — dados_brutos tem colunas simples
        if 'Adj Close' in dados_brutos.columns:
            return dados_brutos[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
        elif 'Close' in dados_brutos.columns:
            return dados_brutos[['Close']].rename(columns={'Close': tickers[0]})
        else:
            raise ValueError("Coluna 'Adj Close' ou 'Close' não encontrada nos dados.")

def otimizar_carteira_sharpe(tickers, min_pct=0.05, max_pct=0.20):
    dados = obter_preco_diario_ajustado(tickers)
    retornos = dados.pct_change().dropna()

    medias = retornos.mean() * 252
    cov = LedoitWolf().fit(retornos).covariance_

    n = len(tickers)

    def sharpe_neg(pesos):
        retorno_esperado = np.dot(pesos, medias)
        volatilidade = np.sqrt(np.dot(pesos.T, np.dot(cov, pesos)))
        return -retorno_esperado / volatilidade

    init = np.array([1/n] * n)
    bounds = tuple((min_pct, max_pct) for _ in range(n))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    resultado = minimize(sharpe_neg, init, bounds=bounds, constraints=constraints)
    if resultado.success:
        return resultado.x
    else:
        raise ValueError("Otimização falhou.")

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
            pesos = otimizar_carteira_sharpe(tickers_validos)
            df_resultado = pd.DataFrame(ativos_validos)
            df_resultado["Alocação (%)"] = (pesos * 100).round(2)
            df_resultado = df_resultado.sort_values("Alocação (%)", ascending=False)
            st.success("✅ Carteira otimizada com Sharpe máximo (restrições padrão: 5%-20%).")
            st.dataframe(df_resultado[["ticker", "setor", "preco_atual", "preco_alvo", "Alocação (%)"]])
        except Exception as e:
            st.error(f"Erro na otimização: {str(e)}")

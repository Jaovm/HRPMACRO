import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.hierarchical_portfolio import HierarchicalRiskParity
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.objective_functions import negative_sharpe

st.set_page_config(page_title="Alocação HRP + Estratégias", layout="wide")
st.title("📈 Alocação com HRP + Estratégias Otimizadas")

# Carteira base
tickers = [
    "AGRO3.SA", "BBAS3.SA", "BBSE3.SA", "BPAC11.SA", "EGIE3.SA",
    "ITUB3.SA", "PRIO3.SA", "PSSA3.SA", "SAPR3.SA", "SBSP3.SA",
    "VIVT3.SA", "WEGE3.SA", "TOTS3.SA", "B3SA3.SA", "TAEE3.SA"
]

st.sidebar.header("📊 Parâmetros de Simulação")
start_date = st.sidebar.date_input("Data inicial", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("Data final", pd.to_datetime("2024-12-31"))

@st.cache_data
def carregar_dados(tickers, start_date, end_date):
    dados = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
    dados = dados.dropna(axis=1)
    retornos = dados.pct_change().dropna()
    return dados, retornos

precos, retornos = carregar_dados(tickers, start_date, end_date)
media_retornos = mean_historical_return(precos)
matriz_cov = CovarianceShrinkage(precos).ledoit_wolf()

# Funções de alocação
def alocacao_hrp(returns):
    hrp = HierarchicalRiskParity()
    hrp.allocate(returns=returns)
    return hrp.clean_weights

def alocacao_hrp_sharpe(returns, media_ret, cov_matrix):
    hrp = HierarchicalRiskParity()
    hrp.allocate(returns=returns)
    tickers_hrp = list(hrp.clean_weights.keys())
    
    ef = EfficientFrontier(media_ret.loc[tickers_hrp], cov_matrix.loc[tickers_hrp, tickers_hrp])
    pesos = ef.max_sharpe()
    return ef.clean_weights()

def alocacao_hrp_maior_retorno(returns, media_ret, cov_matrix):
    hrp = HierarchicalRiskParity()
    hrp.allocate(returns=returns)
    tickers_hrp = list(hrp.clean_weights.keys())

    ef = EfficientFrontier(media_ret.loc[tickers_hrp], cov_matrix.loc[tickers_hrp, tickers_hrp])
    ef.max_quadratic_utility()
    return ef.clean_weights()

def alocacao_hrp_menor_risco(returns, media_ret, cov_matrix):
    hrp = HierarchicalRiskParity()
    hrp.allocate(returns=returns)
    tickers_hrp = list(hrp.clean_weights.keys())

    ef = EfficientFrontier(media_ret.loc[tickers_hrp], cov_matrix.loc[tickers_hrp, tickers_hrp])
    ef.min_volatility()
    return ef.clean_weights()

# Seção de múltiplos cenários macroeconômicos
st.header("🌐 Cenários Macroeconômicos Atuais")
cenarios = {
    "Inflação em alta": ["Setores defensivos", "Utilidades públicas", "Energia"],
    "Inflação em queda": ["Consumo discricionário", "Tecnologia"],
    "Juros altos": ["Utilities", "Elétricas"],
    "Juros baixos": ["Construção civil", "Financeiras"],
    "PIB acelerando": ["Indústria", "Varejo", "Commodities"],
    "PIB desacelerando": ["Saúde", "Serviços essenciais"]
}
for titulo, setores in cenarios.items():
    st.markdown(f"**{titulo}** ➤ {', '.join(setores)}")

# Resultado das alocações
st.header("⚖️ Alocações Sugeridas com Base nas Estratégias")

def exibir_pesos(nome_estrategia, pesos):
    st.subheader(f"📌 {nome_estrategia}")
    df = pd.DataFrame(pesos.items(), columns=["Ativo", "Peso (%)"])
    df["Peso (%)"] = df["Peso (%)"] * 100
    df = df[df["Peso (%)"] > 0.01]
    st.dataframe(df.set_index("Ativo").style.format("{:.2f}"))

exibir_pesos("HRP Puro", alocacao_hrp(retornos))
exibir_pesos("HRP + Sharpe", alocacao_hrp_sharpe(retornos, media_retornos, matriz_cov))
exibir_pesos("HRP + Maior Retorno", alocacao_hrp_maior_retorno(retornos, media_retornos, matriz_cov))
exibir_pesos("HRP + Menor Risco", alocacao_hrp_menor_risco(retornos, media_retornos, matriz_cov))

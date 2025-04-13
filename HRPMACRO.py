import streamlit as st
import pandas as pd
import yfinance as yf
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.efficient_frontier import EfficientFrontier

st.set_page_config(page_title="Alocação HRP + Estratégias", layout="wide")
st.title("📈 Alocação com HRP + Estratégias Otimizadas")

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
    dados = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if 'Adj Close' in data.columns:
                dados[ticker] = data['Adj Close']
            elif 'Close' in data.columns:
                dados[ticker] = data['Close']
        except Exception as e:
            st.warning(f"Erro ao carregar {ticker}: {e}")
            continue

    if not dados:
        return pd.DataFrame(), pd.DataFrame()

    df_dados = pd.DataFrame(dados)
    df_dados = df_dados.fillna(method='ffill').fillna(method='bfill')
    retornos = df_dados.pct_change().dropna()
    return df_dados, retornos

precos, retornos = carregar_dados(tickers, start_date, end_date)
if retornos.empty:
    st.error("Não há dados suficientes para calcular a alocação de portfólio.")
    st.stop()

media_retornos = mean_historical_return(precos)
matriz_cov = CovarianceShrinkage(precos).ledoit_wolf()

def alocacao_hrp(returns):
    cov = returns.cov()
    hrp = HRPOpt(returns=returns, cov_matrix=cov)
    pesos = hrp.optimize()
    return pesos

def alocacao_hrp_sharpe(returns, media_ret, cov_matrix):
    hrp = HRPOpt(returns=returns, cov_matrix=cov_matrix)
    tickers_hrp = list(hrp.optimize().keys())
    ef = EfficientFrontier(media_ret.loc[tickers_hrp], cov_matrix.loc[tickers_hrp, tickers_hrp])
    ef.max_sharpe(risk_free_rate=0.03)
    return ef.clean_weights()

def alocacao_hrp_maior_retorno(returns, media_ret, cov_matrix):
    hrp = HRPOpt(returns=returns, cov_matrix=cov_matrix)
    tickers_hrp = list(hrp.optimize().keys())
    ef = EfficientFrontier(media_ret.loc[tickers_hrp], cov_matrix.loc[tickers_hrp, tickers_hrp])
    ef.max_quadratic_utility()
    return ef.clean_weights()

def alocacao_hrp_menor_risco(returns, media_ret, cov_matrix):
    hrp = HRPOpt(returns=returns, cov_matrix=cov_matrix)
    tickers_hrp = list(hrp.optimize().keys())
    ef = EfficientFrontier(media_ret.loc[tickers_hrp], cov_matrix.loc[tickers_hrp, tickers_hrp])
    ef.min_volatility()
    return ef.clean_weights()

# Cenários econômicos
st.header("🌐 Cenários Macroeconômicos Atuais")

cenarios = {
    "Inflação em alta": ["Setores defensivos", "Utilidades públicas", "Energia"],
    "Inflação em queda": ["Consumo discricionário", "Tecnologia"],
    "Juros altos": ["Utilities", "Elétricas"],
    "Juros baixos": ["Construção civil", "Financeiras"],
    "PIB acelerando": ["Indústria", "Varejo", "Commodities"],
    "PIB desacelerando": ["Saúde", "Serviços essenciais"],
    "Dólar em alta": ["Exportadoras", "Commodities"],
    "Petróleo em alta": ["Petrolíferas", "Energia"]
}

cenarios_selecionados = []
st.sidebar.subheader("🧭 Selecione o cenário atual")
for nome, setores in cenarios.items():
    if st.sidebar.checkbox(nome):
        cenarios_selecionados.append((nome, setores))

if cenarios_selecionados:
    st.markdown("### 🔍 Setores recomendados com base no cenário atual:")
    for nome, setores in cenarios_selecionados:
        st.markdown(f"**{nome}** ➤ {', '.join(setores)}")

# Links para APIs
st.markdown("---")
st.markdown("### 🔗 Links úteis para dados macroeconômicos:")
st.markdown("[📊 API do Banco Central do Brasil (SGS)](https://dadosabertos.bcb.gov.br/dataset)")
st.markdown("[📈 API do IBGE (SIDRA)](https://servicodados.ibge.gov.br/api/docs)")

# Resultados
st.header("⚖️ Alocações Sugeridas com Base nas Estratégias")

def exibir_pesos(nome_estrategia, pesos):
    st.subheader(f"📌 {nome_estrategia}")
    df = pd.DataFrame(list(pesos.items()), columns=["Ativo", "Peso (%)"])
    df["Peso (%)"] = df["Peso (%)"] * 100
    df = df[df["Peso (%)"] > 0.01]
    st.dataframe(df.set_index("Ativo").style.format("{:.2f}"))

exibir_pesos("HRP Puro", alocacao_hrp(retornos))
exibir_pesos("HRP + Sharpe", alocacao_hrp_sharpe(retornos, media_retornos, matriz_cov))
exibir_pesos("HRP + Maior Retorno", alocacao_hrp_maior_retorno(retornos, media_retornos, matriz_cov))
exibir_pesos("HRP + Menor Risco", alocacao_hrp_menor_risco(retornos, media_retornos, matriz_cov))

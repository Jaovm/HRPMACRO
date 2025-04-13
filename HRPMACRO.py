import streamlit as st
import pandas as pd
import yfinance as yf
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.efficient_frontier import EfficientFrontier

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
    dados = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if 'Adj Close' in data.columns:
                dados[ticker] = data['Adj Close']
        except Exception as e:
            st.warning(f"Erro ao carregar dados para {ticker}: {e}")
    
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

# Funções de alocação
def alocacao_hrp(returns):
    cov = returns.cov()
    hrp = HRPOpt(returns=returns, cov_matrix=cov)
    return hrp.optimize()

def alocacao_hrp_sharpe(returns, media_ret, cov_matrix):
    hrp = HRPOpt(returns=returns, cov_matrix=cov_matrix)
    pesos_hrp = hrp.optimize()
    ef = EfficientFrontier(media_ret.loc[list(pesos_hrp.keys())], cov_matrix.loc[list(pesos_hrp.keys()), list(pesos_hrp.keys())])
    return ef.clean_weights()

def alocacao_hrp_maior_retorno(returns, media_ret, cov_matrix):
    hrp = HRPOpt(returns=returns, cov_matrix=cov_matrix)
    pesos_hrp = hrp.optimize()
    ef = EfficientFrontier(media_ret.loc[list(pesos_hrp.keys())], cov_matrix.loc[list(pesos_hrp.keys()), list(pesos_hrp.keys())])
    ef.max_quadratic_utility()
    return ef.clean_weights()

def alocacao_hrp_menor_risco(returns, media_ret, cov_matrix):
    hrp = HRPOpt(returns=returns, cov_matrix=cov_matrix)
    pesos_hrp = hrp.optimize()
    ef = EfficientFrontier(media_ret.loc[list(pesos_hrp.keys())], cov_matrix.loc[list(pesos_hrp.keys()), list(pesos_hrp.keys())])
    ef.min_volatility()
    return ef.clean_weights()

# Seção de múltiplos cenários macroeconômicos
st.header("🌐 Cenários Macroeconômicos Atuais")
with st.expander("Selecionar Cenário Macroeconômico Atual"):
    inflacao = st.checkbox("Inflação em alta")
    juros = st.checkbox("Juros altos")
    pib = st.checkbox("PIB acelerando")
    dolar = st.checkbox("Dólar em alta")
    petroleo = st.checkbox("Petróleo em alta")

    setores_sugeridos = []
    if inflacao:
        setores_sugeridos += ["Utilidades públicas", "Energia", "Alimentos"]
    if juros:
        setores_sugeridos += ["Elétricas", "Telecom", "Utilities"]
    if pib:
        setores_sugeridos += ["Construção", "Varejo", "Industrial"]
    if dolar:
        setores_sugeridos += ["Exportadoras", "Papel e Celulose", "Mineração"]
    if petroleo:
        setores_sugeridos += ["Petróleo e Gás", "Energia"]

    setores_sugeridos = list(set(setores_sugeridos))
    if setores_sugeridos:
        st.success("**Setores recomendados com base no cenário atual:**")
        st.write(", ".join(setores_sugeridos))
    else:
        st.info("Selecione ao menos um cenário macroeconômico para ver sugestões.")

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

# Links de APIs
st.markdown("---")
st.markdown("🔗 **APIs utilizadas para dados macroeconômicos**")
st.markdown("- [API Bacen SGS](https://dadosabertos.bcb.gov.br/dataset/series-temporais)")
st.markdown("- [API IBGE](https://servicodados.ibge.gov.br/api/docs/)")

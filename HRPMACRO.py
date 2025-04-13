import streamlit as st
import pandas as pd
import yfinance as yf
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.efficient_frontier import EfficientFrontier
import requests

st.set_page_config(page_title="Alocação HRP + Estratégias", layout="wide")
st.title("📈 Alocação com HRP + Estratégias Otimizadas")

# Função para obter dados econômicos da API
def obter_dados_economicos():
    # Exemplo de URL fictícia, substitua com as APIs reais que você tiver acesso
    url_ipca = "https://api.ibge.gov.br/ipca"  # Exemplo de URL para IPCA
    url_pib = "https://api.ibge.gov.br/pib"  # Exemplo de URL para PIB
    url_selic = "https://api.bcb.gov.br/selic"  # Exemplo de URL para Selic
    
    try:
        ipca_data = requests.get(url_ipca).json()  # A API para o IPCA
        pib_data = requests.get(url_pib).json()  # A API para o PIB
        selic_data = requests.get(url_selic).json()  # A API para o Selic
        
        ipca = ipca_data["valor"][-1]  # Último valor do IPCA
        pib = pib_data["valor"][-1]  # Último valor do PIB
        selic = selic_data["valor"][-1]  # Última taxa de Selic
        
        return ipca, pib, selic
    except Exception as e:
        st.error(f"Erro ao obter dados econômicos: {e}")
        return None, None, None

# Função para detectar o cenário econômico com base nos dados
def detectar_cenario(ipca, pib, selic):
    cenarios = []
    
    # Inflação
    if ipca > 6:  # Exemplo de inflação alta
        cenarios.append("Inflação em alta")
    elif ipca < 3:  # Exemplo de inflação baixa
        cenarios.append("Inflação em queda")
    
    # Taxa de juros
    if selic > 10:  # Juros altos
        cenarios.append("Juros altos")
    elif selic < 5:  # Juros baixos
        cenarios.append("Juros baixos")
    
    # PIB
    if pib > 0:  # PIB crescendo
        cenarios.append("PIB acelerando")
    else:  # PIB em queda
        cenarios.append("PIB desacelerando")
    
    return cenarios

# Carregar dados econômicos
ipca, pib, selic = obter_dados_economicos()

# Verificar se conseguimos os dados
if ipca is not None and pib is not None and selic is not None:
    cenario_atual = detectar_cenario(ipca, pib, selic)
    st.sidebar.header("🌐 Cenário Macroeconômico Atual (Automático)")
    st.sidebar.write(f"Cenários detectados: {', '.join(cenario_atual)}")
else:
    st.sidebar.warning("Não foi possível obter dados econômicos, por favor, insira manualmente os cenários.")

# Sidebar
st.sidebar.header("📊 Parâmetros de Simulação")
start_date = st.sidebar.date_input("Data inicial", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("Data final", pd.to_datetime("2024-12-31"))

tickers = [
    "AGRO3.SA", "BBAS3.SA", "BBSE3.SA", "BPAC11.SA", "EGIE3.SA",
    "ITUB3.SA", "PRIO3.SA", "PSSA3.SA", "SAPR3.SA", "SBSP3.SA",
    "VIVT3.SA", "WEGE3.SA", "TOTS3.SA", "B3SA3.SA", "TAEE3.SA"
]

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
            else:
                continue
        except:
            continue
    if not dados:
        return pd.DataFrame(), pd.DataFrame()
    df_dados = pd.DataFrame(dados)
    df_dados = df_dados.fillna(method='ffill').fillna(method='bfill')
    retornos = df_dados.pct_change().dropna()
    return df_dados, retornos

precos, retornos = carregar_dados(tickers, start_date, end_date)

if not retornos.empty:
    media_retornos = mean_historical_return(precos)
    matriz_cov = CovarianceShrinkage(precos).ledoit_wolf()

    def alocacao_hrp(returns):
        cov = returns.cov()
        hrp = HRPOpt(returns=returns, cov_matrix=cov)
        return hrp.optimize()

    def alocacao_hrp_sharpe(returns, media_ret, cov_matrix):
        hrp = HRPOpt(returns=returns, cov_matrix=cov_matrix)
        pesos_hrp = hrp.optimize()
        tickers_hrp = list(pesos_hrp.keys())
        ef = EfficientFrontier(media_ret.loc[tickers_hrp], cov_matrix.loc[tickers_hrp, tickers_hrp])
        ef.max_sharpe(risk_free_rate=0.03)
        return ef.clean_weights()

    def alocacao_hrp_maior_retorno(returns, media_ret, cov_matrix):
        hrp = HRPOpt(returns=returns, cov_matrix=cov_matrix)
        pesos_hrp = hrp.optimize()
        tickers_hrp = list(pesos_hrp.keys())
        ef = EfficientFrontier(media_ret.loc[tickers_hrp], cov_matrix.loc[tickers_hrp, tickers_hrp])
        ef.max_quadratic_utility()
        return ef.clean_weights()

    def alocacao_hrp_menor_risco(returns, media_ret, cov_matrix):
        hrp = HRPOpt(returns=returns, cov_matrix=cov_matrix)
        pesos_hrp = hrp.optimize()
        tickers_hrp = list(pesos_hrp.keys())
        ef = EfficientFrontier(media_ret.loc[tickers_hrp], cov_matrix.loc[tickers_hrp, tickers_hrp])
        ef.min_volatility()
        return ef.clean_weights()

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
else:
    st.error("Não há dados suficientes para calcular a alocação de portfólio.")

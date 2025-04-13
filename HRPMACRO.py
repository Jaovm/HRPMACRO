import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.efficient_frontier import EfficientFrontier
import time

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
    # Tentar fazer o download dos dados
    dados = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if 'Adj Close' in data.columns:
                dados[ticker] = data['Adj Close']
            else:
                st.warning(f"Coluna 'Adj Close' não encontrada para {ticker}. Tentando com 'Close'.")
                dados[ticker] = data['Close']
        except Exception as e:
            st.warning(f"Falha ao baixar dados de {ticker}: {e}")
            time.sleep(2)  # Espera 2 segundos antes de tentar novamente
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                if 'Adj Close' in data.columns:
                    dados[ticker] = data['Adj Close']
                else:
                    dados[ticker] = data['Close']
            except Exception as e:
                st.warning(f"Falha ao tentar novamente baixar dados de {ticker}: {e}")
                continue

    # Verificar se o dicionário de dados está vazio
    if not dados:
        st.error("Não foi possível baixar dados para nenhum ativo.")
        return pd.DataFrame(), pd.DataFrame()

    # Verificar se algum valor em `dados` é válido
    if all(v is None or len(v) == 0 for v in dados.values()):
        st.error("Os dados baixados estão vazios. Não há informações suficientes.")
        return pd.DataFrame(), pd.DataFrame()

    # Se os dados são válidos, criar DataFrame
    try:
        df_dados = pd.DataFrame(dados)
    except Exception as e:
        st.error(f"Erro ao criar DataFrame a partir dos dados: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # Verificar se o DataFrame resultante está vazio
    if df_dados.empty:
        st.error("O DataFrame resultante está vazio. Não há dados suficientes para calcular a alocação.")
        return pd.DataFrame(), pd.DataFrame()

    df_dados = df_dados.dropna(axis=1)  # Remove ativos com dados ausentes
    df_dados = df_dados.fillna(method='ffill').fillna(method='bfill')  # Preenche NaNs
    retornos = df_dados.pct_change().dropna()  # Calcula os retornos percentuais
    return df_dados, retornos

precos, retornos = carregar_dados(tickers, start_date, end_date)
if retornos.empty:
    st.error("Não há dados suficientes para calcular a alocação de portfólio.")
else:
    try:
        media_retornos = mean_historical_return(precos)
    except Exception as e:
        st.error(f"Erro ao calcular a média de retornos: {e}")
    
    # Garantir que a matriz de covariância não contenha NaN
    try:
        matriz_cov = CovarianceShrinkage(precos).ledoit_wolf()  # Cálculo da matriz de covariância
    except ValueError as e:
        st.error(f"Erro ao calcular a matriz de covariância: {e}")
    
    # Funções de alocação
    def alocacao_hrp(returns):
        cov = returns.cov()
        hrp = HRPOpt(returns=returns, cov_matrix=cov)
        pesos = hrp.optimize()
        return pesos

    def alocacao_hrp_sharpe(returns, media_ret, cov_matrix):
        hrp = HRPOpt(returns=returns, cov_matrix=cov_matrix)
        pesos_hrp = hrp.optimize()
        tickers_hrp = list(pesos_hrp.keys())
        
        ef = EfficientFrontier(media_ret.loc[tickers_hrp], cov_matrix.loc[tickers_hrp, tickers_hrp])
        pesos_sharpe = ef.max_sharpe(risk_free_rate=0.03)  # Definindo uma taxa livre de risco
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

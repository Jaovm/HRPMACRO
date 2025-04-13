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
    df_dados = pd.DataFrame()
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            preco = None
            if 'Adj Close' in data.columns:
                preco = data['Adj Close']
            elif 'Close' in data.columns:
                preco = data['Close']
            if preco is not None:
                df_dados[ticker] = preco
        except Exception as e:
            st.warning(f"Erro ao baixar {ticker}: {e}")
    df_dados = df_dados.dropna(axis=1)
    df_dados = df_dados.fillna(method='ffill').fillna(method='bfill')
    retornos = df_dados.pct_change().dropna()
    return df_dados, retornos

precos, retornos = carregar_dados(tickers, start_date, end_date)
if retornos.empty:
    st.error("Não há dados suficientes para calcular a alocação de portfólio.")
else:
    try:
        media_retornos = mean_historical_return(precos)
        matriz_cov = CovarianceShrinkage(precos).ledoit_wolf()
    except Exception as e:
        st.error(f"Erro nos cálculos: {e}")

    def alocacao_hrp(returns):
        cov = returns.cov()
        hrp = HRPOpt(returns=returns, cov_matrix=cov)
        return hrp.optimize()

    def alocacao_hrp_sharpe(returns, media_ret, cov_matrix):
        hrp = HRPOpt(returns=returns, cov_matrix=cov_matrix)
        pesos_hrp = hrp.optimize()
        ef = EfficientFrontier(media_ret.loc[list(pesos_hrp.keys())], cov_matrix.loc[list(pesos_hrp.keys()), list(pesos_hrp.keys())])
        ef.max_sharpe(risk_free_rate=0.03)
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

    # Seção macroeconômica com seleção
    st.header("🌐 Cenários Macroeconômicos Atuais")

    cenarios_macro = {
        "Inflação em alta": ["Setores defensivos", "Utilidades públicas", "Energia"],
        "Inflação em queda": ["Consumo discricionário", "Tecnologia"],
        "Juros altos": ["Utilities", "Elétricas"],
        "Juros baixos": ["Construção civil", "Financeiras"],
        "PIB acelerando": ["Indústria", "Varejo", "Commodities"],
        "PIB desacelerando": ["Saúde", "Serviços essenciais"]
    }

    st.subheader("🧭 Selecione o(s) cenário(s) macroeconômico(s) atual(is)")
    cenarios_selecionados = []
    for nome in cenarios_macro:
        if st.checkbox(nome):
            cenarios_selecionados.append(nome)

    setores_recomendados = set()
    for cenario in cenarios_selecionados:
        setores_recomendados.update(cenarios_macro[cenario])

    if setores_recomendados:
        st.markdown(f"### 🎯 Setores recomendados: {', '.join(setores_recomendados)}")

        setores_ativos = {
            "AGRO3.SA": "Commodities",
            "BBAS3.SA": "Financeiras",
            "BBSE3.SA": "Financeiras",
            "BPAC11.SA": "Financeiras",
            "EGIE3.SA": "Utilidades públicas",
            "ITUB3.SA": "Financeiras",
            "PRIO3.SA": "Energia",
            "PSSA3.SA": "Financeiras",
            "SAPR3.SA": "Utilidades públicas",
            "SBSP3.SA": "Utilidades públicas",
            "VIVT3.SA": "Serviços essenciais",
            "WEGE3.SA": "Indústria",
            "TOTS3.SA": "Tecnologia",
            "B3SA3.SA": "Financeiras",
            "TAEE3.SA": "Utilidades públicas"
        }

        ativos_recomendados = [ticker for ticker, setor in setores_ativos.items() if setor in setores_recomendados]

        if ativos_recomendados:
            st.success("✅ Empresas da sua carteira que se beneficiam do cenário atual:")
            st.write(", ".join(ativos_recomendados))
        else:
            st.warning("Nenhum ativo da sua carteira atual pertence aos setores favorecidos.")
    else:
        st.info("Selecione um cenário acima para ver recomendações.")

    # Resultados das estratégias
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

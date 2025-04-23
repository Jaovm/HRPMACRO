import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from collections import defaultdict


def validar_tickers(tickers):
    """Valida os tickers para garantir que possuem dados básicos disponíveis."""
    validos = []
    for ticker in tickers:
        try:
            dados = yf.Ticker(ticker).history(period="1d")
            if not dados.empty:
                validos.append(ticker)
            else:
                st.warning(f"⚠️ Ticker sem dados: {ticker}")
        except Exception as e:
            st.warning(f"⚠️ Erro ao validar ticker {ticker}: {e}")
    return validos


def baixar_dados(tickers, period="2y", interval="1mo"):
    """Baixa dados do Yahoo Finance para os tickers fornecidos."""
    try:
        dados = yf.download(tickers, period=period, interval=interval, group_by="ticker", auto_adjust=True)
        if dados.empty:
            st.warning(f"⚠️ Nenhum dado retornado para os tickers: {tickers}")
        return dados
    except Exception as e:
        st.error(f"⚠️ Erro ao baixar dados para os tickers {tickers}: {e}")
        return pd.DataFrame()


def gerar_dados_simulados(tickers, periodos=24):
    """Gera dados simulados para tickers sem dados válidos."""
    st.warning("⚠️ Gerando dados simulados para os tickers ausentes...")
    datas = pd.date_range(end=pd.Timestamp.today(), periods=periodos, freq='MS')
    dados_simulados = {ticker: np.random.normal(0, 0.02, len(datas)) for ticker in tickers}
    return pd.DataFrame(dados_simulados, index=datas)


def obter_dados_setoriais(setores, tickers_validos):
    """Obtém dados setoriais para cada setor."""
    retornos_setoriais = {}
    for setor, tickers in setores.items():
        tickers_validos_setor = [t for t in tickers if t in tickers_validos]
        if not tickers_validos_setor:
            st.warning(f"⚠️ Nenhum ticker válido para o setor {setor}. Gerando dados simulados.")
            simulados = gerar_dados_simulados(tickers)
            retornos_setoriais[setor] = simulados.mean(axis=1)
            continue

        st.info(f"🔄 Baixando dados para setor: {setor} → {tickers_validos_setor}")
        dados = baixar_dados(tickers_validos_setor)
        if dados.empty:
            st.warning(f"⚠️ Dados vazios para setor {setor}. Gerando dados simulados.")
            simulados = gerar_dados_simulados(tickers)
            retornos_setoriais[setor] = simulados.mean(axis=1)
            continue

        try:
            if 'Adj Close' in dados.columns.get_level_values(0):
                dados = dados['Adj Close']
            retornos = dados.pct_change().dropna()
            retornos_setoriais[setor] = retornos.mean(axis=1)
        except Exception as e:
            st.error(f"⚠️ Erro ao processar dados para setor {setor}: {e}")
            simulados = gerar_dados_simulados(tickers)
            retornos_setoriais[setor] = simulados.mean(axis=1)

    return pd.DataFrame(retornos_setoriais)


def construir_modelo_regressao(retornos_setoriais, macro_data):
    """Constrói modelos de regressão para cada setor."""
    coeficientes = {}
    fatores_macro = macro_data.columns
    for setor in retornos_setoriais.columns:
        try:
            y = retornos_setoriais[setor]
            X = macro_data
            X = sm.add_constant(X)
            modelo = sm.OLS(y, X).fit()
            coeficientes[setor] = modelo.params.drop('const').to_dict()
            st.success(f"✅ Regressão bem-sucedida para setor {setor}")
        except Exception as e:
            st.error(f"⚠️ Erro ao construir regressão para setor {setor}: {e}")
    return coeficientes


def main():
    # Dados macroeconômicos simulados
    datas = pd.date_range(end=pd.Timestamp.today(), periods=24, freq='MS')
    macro_data = pd.DataFrame({
        'selic': np.random.normal(9, 1, len(datas)),
        'ipca': np.random.normal(3, 0.5, len(datas)),
        'dolar': np.random.normal(5.2, 0.3, len(datas)),
        'pib': np.random.normal(2.0, 0.7, len(datas)),
        'commodities_agro': np.random.normal(9, 2, len(datas)),
        'commodities_minerio': np.random.normal(110, 15, len(datas)),
        'commodities_petroleo': np.random.normal(85, 10, len(datas)),
    }, index=datas)

    # Dados setoriais (exemplo)
    setores = {
        'Bancos': ['ITUB4.SA', 'BBDC4.SA', 'SANB11.SA', 'BBAS3.SA'],
        'Seguradoras': ['BBSE3.SA', 'PSSA3.SA', 'CXSE3.SA'],
        'Energia Elétrica': ['EGIE3.SA', 'TAEE11.SA', 'CMIG4.SA'],
    }

    # Validar tickers
    todos_tickers = [ticker for tickers in setores.values() for ticker in tickers]
    tickers_validos = validar_tickers(todos_tickers)

    # Obter dados setoriais
    retornos_setoriais = obter_dados_setoriais(setores, tickers_validos)

    if retornos_setoriais.empty:
        st.error("⚠️ Nenhum dado setorial disponível. Abortando.")
        return

    # Construir modelo de regressão
    coeficientes = construir_modelo_regressao(retornos_setoriais, macro_data)

    # Exibir resultados
    if coeficientes:
        st.info("📈 Coeficientes finais:")
        st.write(pd.DataFrame(coeficientes).T)
    else:
        st.error("⚠️ Nenhum modelo de regressão foi construído.")


if __name__ == "__main__":
    main()

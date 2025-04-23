import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from collections import defaultdict
from dados_setoriais import setores_por_ticker

def baixar_dados_validos(tickers, period="2y", interval="1mo"):
    """
    Baixa dados do Yahoo Finance para uma lista de tickers, retornando apenas aqueles com série histórica válida.
    Sempre retorna DataFrame com colunas = tickers válidos e linhas = datas.
    """
    if not tickers:
        return pd.DataFrame()
    dados = yf.download(tickers, period=period, interval=interval, group_by="ticker", auto_adjust=True, progress=False)
    # Se for MultiIndex, pegar Close
    if isinstance(dados.columns, pd.MultiIndex):
        if "Close" in dados.columns.get_level_values(0):
            dados = dados["Close"]
        elif "Adj Close" in dados.columns.get_level_values(0):
            dados = dados["Adj Close"]
        else:
            return pd.DataFrame()
    # Remove colunas sem dados
    if isinstance(dados, pd.DataFrame):
        dados = dados.dropna(axis=1, how='all')
    return dados

def gerar_dados_macro(periodos=24):
    datas = pd.date_range(end=pd.Timestamp.today(), periods=periodos, freq='MS')
    macro_data = pd.DataFrame({
        'selic': np.random.normal(9, 1, len(datas)),
        'ipca': np.random.normal(3, 0.5, len(datas)),
        'dolar': np.random.normal(5.2, 0.3, len(datas)),
        'pib': np.random.normal(2.0, 0.7, len(datas)),
        'commodities_agro': np.random.normal(9, 2, len(datas)),
        'commodities_minerio': np.random.normal(110, 15, len(datas)),
        'commodities_petroleo': np.random.normal(85, 10, len(datas)),
    }, index=datas)
    return macro_data

def obter_setores_da_carteira(tickers):
    setores_set = set()
    for ticker in tickers:
        setor = setores_por_ticker.get(ticker)
        if setor:
            setores_set.add(setor)
    return setores_set

def obter_retornos_setoriais(setores, n_tickers_setor=3, periodo="2y", intervalo="1mo"):
    """
    Para cada setor, pega até n_tickers_setor tickers válidos, retorna DataFrame com retorno médio mensal por setor.
    """
    retornos_setoriais = {}
    for setor in setores:
        tickers_setor = [t for t, s in setores_por_ticker.items() if s == setor][:n_tickers_setor]
        dados = baixar_dados_validos(tickers_setor, period=periodo, interval=intervalo)
        if dados.empty:
            continue
        dados = dados.ffill().bfill()
        retornos = dados.pct_change().dropna()
        media_mensal = retornos.mean(axis=1)
        retornos_setoriais[setor] = media_mensal
    if not retornos_setoriais:
        return pd.DataFrame()
    retornos_df = pd.DataFrame(retornos_setoriais).dropna()
    return retornos_df

def ajustar_e_sincronizar_macro_com_retornos(macro_data, retornos_df):
    # Ajusta os índices (datas) para garantir merge adequado
    macro_data.index = pd.to_datetime(macro_data.index).normalize()
    retornos_df.index = pd.to_datetime(retornos_df.index).normalize()
    merged = macro_data.join(retornos_df, how='inner').dropna()
    return merged

def normalizar_coeficientes(coef_dict):
    # Normaliza para escala -2 a 2
    return {
        setor: {
            fator: int(np.clip(round(valor * 2), -2, 2)) for fator, valor in coef.items()
        } for setor, coef in coef_dict.items()
    }

def obter_sensibilidade_regressao(tickers_carteira=None, normalizar=True, salvar_csv=False):
    """
    Modelo robusto de regressão setorial.
    1. Identifica setores presentes na carteira (ou todos).
    2. Para cada setor, calcula retorno médio mensal histórico com até 3 tickers válidos.
    3. Obtém histórico macro simulado ou real.
    4. Para cada setor, faz uma regressão múltipla dos retornos setoriais em função dos fatores macro.
    5. Retorna coeficientes normalizados por setor.
    """
    n_periodos = 24
    macro_data = gerar_dados_macro(periodos=n_periodos)
    if tickers_carteira:
        setores = obter_setores_da_carteira(tickers_carteira)
    else:
        setores = set(setores_por_ticker.values())
    retornos_df = obter_retornos_setoriais(setores, n_tickers_setor=3, periodo=f"{n_periodos}mo", intervalo="1mo")
    if retornos_df.empty:
        st.warning("⚠️ Nenhum dado de retorno setorial disponível para regressão setorial.")
        return {}
    merged = ajustar_e_sincronizar_macro_com_retornos(macro_data, retornos_df)
    fatores_macro = ['selic', 'ipca', 'dolar', 'pib', 'commodities_agro', 'commodities_minerio', 'commodities_petroleo']
    coeficientes = {}
    for setor in retornos_df.columns:
        try:
            y = merged[setor]
            X = merged[fatores_macro]
            X = sm.add_constant(X)
            modelo = sm.OLS(y, X).fit()
            coef = modelo.params.drop('const')
            coeficientes[setor] = coef.to_dict()
        except Exception as e:
            st.warning(f"⚠️ Regressão falhou para setor {setor}: {e}")
    if not coeficientes:
        st.warning("⚠️ Nenhum coeficiente foi gerado. Retornando dicionário vazio.")
        return {}
    if normalizar:
        coeficientes = normalizar_coeficientes(coeficientes)
    if salvar_csv:
        pd.DataFrame.from_dict(coeficientes, orient='index').to_csv("sensibilidade_setorial.csv")
    return coeficientes

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from dados_setoriais import setores_por_ticker

def filtrar_tickers_com_dados(tickers, periodo="2y", intervalo="1mo"):
    """Retorna apenas tickers com histórico válido (Close ou Adj Close) no Yahoo Finance."""
    tickers_validos = []
    for ticker in tickers:
        try:
            dados = yf.download(ticker, period=periodo, interval=intervalo, progress=False)
            if not dados.empty and (("Close" in dados.columns) or ("Adj Close" in dados.columns)):
                tickers_validos.append(ticker)
        except Exception:
            continue
    return tickers_validos

def baixar_dados_validos(tickers, period="2y", interval="1mo"):
    """Baixa dados do Yahoo Finance para tickers válidos, retorna DataFrame de preços de fechamento."""
    if not tickers:
        return pd.DataFrame()
    dados = yf.download(tickers, period=period, interval=interval, group_by="ticker", auto_adjust=True, progress=False)
    # Se for MultiIndex, pegar Close/Adj Close
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

def gerar_dados_simulados_para_setor(setor, n_periodos=24):
    datas = pd.date_range(end=pd.Timestamp.today(), periods=n_periodos, freq='MS')
    simulados = pd.Series(np.random.normal(0, 0.025, n_periodos), index=datas, name=setor)
    return simulados

def obter_retornos_setoriais(setores, n_tickers_setor=3, periodo="2y", intervalo="1mo", n_periodos=24):
    """
    Para cada setor, tenta até n_tickers_setor com dados válidos, senão gera simulado.
    Sempre retorna DataFrame de retornos médios mensais por setor.
    """
    retornos_setoriais = {}
    for setor in setores:
        tickers_setor = [t for t, s in setores_por_ticker.items() if s == setor]
        tickers_validos = filtrar_tickers_com_dados(tickers_setor, periodo, intervalo)[:n_tickers_setor]
        if not tickers_validos:
            # GERA SIMULADO SE NÃO HOUVER TICKER VÁLIDO
            retornos_setoriais[setor] = gerar_dados_simulados_para_setor(setor, n_periodos)
            continue
        dados = baixar_dados_validos(tickers_validos, period=periodo, interval=intervalo)
        if dados.empty:
            retornos_setoriais[setor] = gerar_dados_simulados_para_setor(setor, n_periodos)
            continue
        dados = dados.ffill().bfill()
        retornos = dados.pct_change().dropna()
        media_mensal = retornos.mean(axis=1)
        retornos_setoriais[setor] = media_mensal
    retornos_df = pd.DataFrame(retornos_setoriais).dropna()
    return retornos_df

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

def obter_setores_a_partir_ativos(ativos_usuario):
    setores_set = set()
    for ticker in ativos_usuario:
        setor = setores_por_ticker.get(ticker)
        if setor:
            setores_set.add(setor)
    return setores_set

def ajustar_e_sincronizar_macro_com_retornos(macro_data, retornos_df):
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
    Modelo robusto de regressão setorial para uso no HRPMACRO.py.
    Sempre retorna dados para todos setores relevantes, usando simulado se necessário.
    Os setores são filtrados a partir dos ativos inseridos pelo usuário.
    """
    n_periodos = 24
    macro_data = gerar_dados_macro(periodos=n_periodos)
    if tickers_carteira:
        setores = obter_setores_a_partir_ativos(tickers_carteira)
    else:
        setores = set(setores_por_ticker.values())
    if not setores:
        st.warning("⚠️ Nenhum setor identificado para os ativos fornecidos.")
        return {}
    retornos_df = obter_retornos_setoriais(setores, n_tickers_setor=3, periodo=f"{n_periodos}mo", intervalo="1mo", n_periodos=n_periodos)
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

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from collections import defaultdict
from dados_setoriais import setores_por_ticker


def obter_sensibilidade_regressao(tickers_carteira=None, normalizar=False, salvar_csv=False):
    # Gerar dados simulados de indicadores macroeconômicos
    datas = pd.date_range(end=pd.Timestamp.today(), periods=24, freq='ME')
    macro_data = pd.DataFrame({
        'data': datas,
        'selic': np.random.normal(9, 1, len(datas)),
        'ipca': np.random.normal(3, 0.5, len(datas)),
        'dolar': np.random.normal(5.2, 0.3, len(datas)),
        'pib': np.random.normal(2.0, 0.7, len(datas)),
        'commodities_agro': np.random.normal(9, 2, len(datas)),
        'commodities_minerio': np.random.normal(110, 15, len(datas)),
        'commodities_petroleo': np.random.normal(85, 10, len(datas)),
    })
    macro_data.set_index('data', inplace=True)

    # Organização de tickers por setor
    setores = defaultdict(list)
    for ticker, setor in setores_por_ticker.items():
        setores[setor].append(ticker)

    # Filtrar setores com base nos tickers da carteira
    if tickers_carteira:
        setores_ativos = {setores_por_ticker[t] for t in tickers_carteira if t in setores_por_ticker}
        st.info(f"📌 Setores ativos identificados: {setores_ativos}")
        setores = {s: setores[s][:3] for s in setores_ativos if s in setores}
    else:
        setores = {s: tks[:3] for s, tks in setores.items()}

    st.info(f"📌 Setores selecionados para regressão: {list(setores.keys())}")

    # Obter retornos setoriais
    retornos_setoriais = {}
    for setor, tickers in setores.items():
        try:
            st.info(f"🔄 Baixando dados para setor: {setor} -> {tickers}")
            dados = yf.download(tickers, period="2y", interval="1mo", group_by="ticker", auto_adjust=True)

            if isinstance(dados.columns, pd.MultiIndex):
                dados = dados['Close']
            elif 'Close' in dados:
                dados = dados[['Close']]
            else:
                st.warning(f"⚠️ Nenhuma coluna 'Close' encontrada para {setor}")
                continue

            dados = dados.fillna(method='ffill')
            retornos = dados.pct_change().dropna()
            media_mensal = retornos.mean(axis=1)
            retornos_setoriais[setor] = media_mensal
        except Exception as e:
            st.error(f"⚠️ Erro ao processar setor {setor}: {e}")
            continue

    # Verificar se há dados de retorno
    if not retornos_setoriais:
        st.error("⚠️ Nenhum dado de retorno setorial disponível.")
        return {}

    retornos_df = pd.DataFrame(retornos_setoriais).dropna()
    st.info(f"📈 Retornos setoriais disponíveis: {list(retornos_df.columns)}")

    # Mesclar dados macroeconômicos com retornos setoriais
    dados_merged = macro_data.join(retornos_df, how='inner').dropna()

    coeficientes = {}
    fatores_macro = ['selic', 'ipca', 'dolar', 'pib', 'commodities_agro', 'commodities_minerio', 'commodities_petroleo']
    for setor in retornos_df.columns:
        try:
            # Regressão de fatores macroeconômicos contra retornos setoriais
            y = dados_merged[setor]
            X = dados_merged[fatores_macro]
            X = sm.add_constant(X)
            modelo = sm.OLS(y, X).fit()
            coef = modelo.params.drop('const')
            coeficientes[setor] = coef.to_dict()
            st.success(f"✅ Regressão bem-sucedida para setor {setor}")
        except Exception as e:
            st.error(f"⚠️ Regressão falhou para setor {setor}: {e}")
            continue

    # Verificar se há coeficientes gerados
    if not coeficientes:
        st.error("⚠️ Nenhum coeficiente foi gerado. Retornando dicionário vazio.")
        return {}

    st.info(f"📈 Coeficientes finais: {coeficientes}")

    # Normalizar coeficientes, se necessário
    if normalizar:
        coeficientes = normalizar_coeficientes(coeficientes)

    # Salvar coeficientes em CSV, se configurado
    if salvar_csv:
        try:
            pd.DataFrame.from_dict(coeficientes, orient='index').to_csv("sensibilidade_setorial.csv")
            st.success("✅ Coeficientes salvos em sensibilidade_setorial.csv")
        except Exception as e:
            st.error(f"⚠️ Falha ao salvar coeficientes em CSV: {e}")

    st.success(f"✅ Coeficientes gerados para {len(coeficientes)} setores.")
    return coeficientes


def normalizar_coeficientes(coef_dict):
    """Normalizar coeficientes para escala fixa."""
    return {
        setor: {
            fator: int(np.clip(round(valor * 2), -2, 2)) for fator, valor in coef.items()
        } for setor, coef in coef_dict.items()
    }

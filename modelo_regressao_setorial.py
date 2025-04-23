import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from collections import defaultdict
from dados_setoriais import setores_por_ticker

def obter_sensibilidade_regressao(tickers_carteira=None, normalizar=False, salvar_csv=False):
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

    setores = defaultdict(list)
    for ticker, setor in setores_por_ticker.items():
        setores[setor].append(ticker)

    if tickers_carteira:
        setores_ativos = {setores_por_ticker[t] for t in tickers_carteira if t in setores_por_ticker}
        print("📌 Setores ativos identificados:", setores_ativos)
        setores = {s: setores[s][:3] for s in setores_ativos if s in setores}
    else:
        setores = {s: tks[:3] for s, tks in setores.items()}

    print("📌 Setores selecionados para regressão:", list(setores.keys()))

    retornos_setoriais = {}
    for setor, tickers in setores.items():
        try:
            print(f"🔄 Baixando dados para setor: {setor} -> {tickers}")
            dados = yf.download(tickers, period="2y", interval="1mo", group_by="ticker", auto_adjust=True)
            if isinstance(dados.columns, pd.MultiIndex):
                dados = dados['Close']
            elif 'Close' in dados:
                dados = dados[['Close']]
            else:
                print(f"⚠️ Nenhuma coluna 'Close' encontrada para {setor}")
                continue

            dados = dados.fillna(method='ffill')
            retornos = dados.pct_change().dropna()
            media_mensal = retornos.mean(axis=1)
            retornos_setoriais[setor] = media_mensal
        except Exception as e:
            print(f"⚠️ Erro ao processar setor {setor}: {e}")
            continue

    if not retornos_setoriais:
        print("⚠️ Nenhum dado de retorno setorial disponível.")
        return {}

    retornos_df = pd.DataFrame(retornos_setoriais).dropna()
    print("📈 Retornos setoriais disponíveis:", list(retornos_df.columns))

    dados_merged = macro_data.join(retornos_df, how='inner').dropna()

    coeficientes = {}
    fatores_macro = ['selic', 'ipca', 'dolar', 'pib', 'commodities_agro', 'commodities_minerio', 'commodities_petroleo']
    for setor in retornos_df.columns:
        try:
            y = dados_merged[setor]
            X = dados_merged[fatores_macro]
            X = sm.add_constant(X)
            modelo = sm.OLS(y, X).fit()
            coef = modelo.params.drop('const')
            coeficientes[setor] = coef.to_dict()
            print(f"✅ Regressão bem-sucedida para {setor}")
        except Exception as e:
            print(f"⚠️ Regressão falhou para setor {setor}: {e}")
            continue

    if not coeficientes:
        print("⚠️ Nenhum coeficiente foi gerado. Retornando dicionário vazio.")
        return {}

    print("📈 Coeficientes finais:", coeficientes)

    if normalizar:
        coeficientes = normalizar_coeficientes(coeficientes)

    if salvar_csv:
        pd.DataFrame.from_dict(coeficientes, orient='index').to_csv("sensibilidade_setorial.csv")

    print(f"✅ Coeficientes gerados para {len(coeficientes)} setores.")
    return coeficientes

def normalizar_coeficientes(coef_dict):
    return {
        setor: {
            fator: int(np.clip(round(valor * 2), -2, 2)) for fator, valor in coef.items()
        } for setor, coef in coef_dict.items()
    }

from modelo_regressao_setorial import obter_sensibilidade_regressao

carteira = dict(zip(tickers, pesos_atuais))
tickers_carteira = list(carteira.keys())

sensibilidade_setorial = obter_sensibilidade_regressao(
    tickers_carteira=tickers_carteira,
    normalizar=True,
    salvar_csv=True
)

if sensibilidade_setorial:
    with st.expander("📉 Ver Sensibilidade Setorial (Regressão)"):
        st.json(sensibilidade_setorial)
else:
    st.warning("⚠️ Nenhum dado de sensibilidade disponível.")

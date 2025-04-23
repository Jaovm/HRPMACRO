import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from collections import defaultdict
from dados_setoriais import setores_por_ticker

def obter_sensibilidade_regressao(normalizar=False, salvar_csv=False):
    datas = pd.date_range(end=pd.Timestamp.today(), periods=24, freq='M')
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

    # Usa no máximo 3 ativos por setor
    setores_reduzidos = {k: v[:3] for k, v in setores.items()}

    retornos_setoriais = {}
    for setor, tickers in setores_reduzidos.items():
        try:
            dados = yf.download(tickers, period="2y", interval="1mo", group_by="ticker", auto_adjust=True)

            if isinstance(dados.columns, pd.MultiIndex):
                dados = dados['Close']
            elif 'Close' in dados:
                dados = dados[['Close']]
            else:
                continue

            dados = dados.fillna(method='ffill')
            retornos = dados.pct_change().dropna()
            media_mensal = retornos.mean(axis=1)
            retornos_setoriais[setor] = media_mensal
        except Exception as e:
            print(f"Erro ao processar setor {setor}: {e}")
            continue

    retornos_df = pd.DataFrame(retornos_setoriais).dropna()
    retornos_df.index = pd.to_datetime(retornos_df.index)

    # Sincroniza macro com retornos
    dados_merged = macro_data.join(retornos_df, how='inner').dropna()

    coeficientes = {}
    fatores_macro = ['selic', 'ipca', 'dolar', 'pib', 'commodities_agro', 'commodities_minerio', 'commodities_petroleo']
    for setor in retornos_df.columns:
        y = dados_merged[setor]
        X = dados_merged[fatores_macro]
        X = sm.add_constant(X)
        modelo = sm.OLS(y, X).fit()
        coef = modelo.params.drop('const')
        coeficientes[setor] = coef.to_dict()

    if normalizar:
        coeficientes = normalizar_coeficientes(coeficientes)

    if salvar_csv:
        df_coef = pd.DataFrame.from_dict(coeficientes, orient='index')
        df_coef.to_csv("sensibilidade_setorial.csv")

    return coeficientes


def normalizar_coeficientes(coef_dict):
    """
    Normaliza coeficientes para faixa -2 a 2.
    """
    return {
        setor: {
            fator: int(np.clip(round(valor * 2), -2, 2)) for fator, valor in coef.items()
        } for setor, coef in coef_dict.items()
    }

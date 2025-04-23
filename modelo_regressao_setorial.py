import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from collections import defaultdict
from dados_setoriais import setores_por_ticker




def obter_sensibilidade_regressao():
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


    retornos_setoriais = {}
    for setor, tickers in setores.items():
        dados = yf.download(tickers, period="2y", interval="1mo", group_by="ticker", auto_adjust=True)
            if isinstance(dados.columns, pd.MultiIndex):
                dados = dados.stack(level=0).unstack(level=1)  # reorganiza MultiIndex
                dados = dados['Close'] if 'Close' in dados else dados  # fallback se "Close" for único
            
            elif 'Adj Close' in dados:
                dados = dados['Adj Close']
            elif 'Close' in dados:
                dados = dados['Close']
            else:
                continue  # pular se não tem dados relevantes

        if isinstance(dados, pd.Series):
            dados = dados.to_frame()
        dados = dados.fillna(method='ffill')
        retornos = dados.pct_change().dropna()
        media_mensal = retornos.mean(axis=1)
        retornos_setoriais[setor] = media_mensal

    retornos_df = pd.DataFrame(retornos_setoriais).dropna()
    retornos_df.index = pd.to_datetime(retornos_df.index)
    dados_merged = macro_data.join(retornos_df, how='inner')

    coeficientes = {}
    fatores_macro = ['selic', 'ipca', 'dolar', 'pib', 'commodities_agro', 'commodities_minerio', 'commodities_petroleo']
    for setor in setores:
        y = dados_merged[setor]
        X = dados_merged[fatores_macro]
        X = sm.add_constant(X)
        modelo = sm.OLS(y, X).fit()
        coef = modelo.params.drop('const')
        coeficientes[setor] = coef.to_dict()

    return coeficientes

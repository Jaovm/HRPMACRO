import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from collections import defaultdict
from dados_setoriais import setores_por_ticker
from time import sleep


def validar_tickers(tickers):
    """Valida os tickers para garantir que possuem dados disponíveis."""
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


def baixar_dados_com_retentativa(tickers, period="2y", interval="1mo", max_retentativas=3):
    """Baixa dados com retentativas em caso de falha."""
    for tentativa in range(max_retentativas):
        try:
            sleep(1)  # Aguarda 1 segundo entre as tentativas para evitar limitações
            return yf.download(tickers, period=period, interval=interval, group_by="ticker", auto_adjust=True)
        except Exception as e:
            st.warning(f"⚠️ Tentativa {tentativa + 1}/{max_retentativas} falhou para tickers {tickers}. ({e})")
            sleep(5)  # Espera antes de tentar novamente
    raise Exception(f"⚠️ Não foi possível baixar os dados para tickers {tickers} após {max_retentativas} tentativas.")


def gerar_dados_simulados(tickers, periodos=24):
    """Gera dados simulados para tickers sem dados válidos."""
    st.warning("⚠️ Gerando dados simulados para os tickers ausentes...")
    datas = pd.date_range(end=pd.Timestamp.today(), periods=periodos, freq='ME')
    dados_simulados = {ticker: np.random.normal(0, 0.02, len(datas)) for ticker in tickers}
    return pd.DataFrame(dados_simulados, index=datas)


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

    # Validação de tickers antes do processamento
    st.info("📋 Validando tickers...")
    tickers_validos = validar_tickers([t for setor in setores.values() for t in setor])
    tickers_invalidos = set([t for setor in setores.values() for t in setor]) - set(tickers_validos)
    if not tickers_validos:
        st.error("❌ Nenhum ticker válido encontrado. Abortando.")
        return {}

    # Filtrar setores com base nos tickers válidos
    setores = {setor: [t for t in tickers if t in tickers_validos] for setor, tickers in setores.items()}
    if tickers_carteira:
        setores = {s: tks for s, tks in setores.items() if any(t in tickers_carteira for t in tks)}

    st.info(f"📌 Setores selecionados para regressão: {list(setores.keys())}")

    retornos_setoriais = {}
    for setor, tickers in setores.items():
        if not tickers:
            st.warning(f"⚠️ Nenhum ticker válido para o setor {setor}. Ignorando.")
            continue

        try:
            st.info(f"🔄 Baixando dados para setor: {setor} → {tickers}")
            dados = baixar_dados_com_retentativa(tickers)

            # Verificar se os dados possuem as colunas esperadas
            if isinstance(dados.columns, pd.MultiIndex):
                if 'Close' in dados.columns.get_level_values(0):
                    dados = dados['Close']
                elif 'Adj Close' in dados.columns.get_level_values(0):
                    dados = dados['Adj Close']
                else:
                    st.warning(f"⚠️ Nenhuma coluna 'Close' ou 'Adj Close' encontrada para {setor}")
                    continue
            elif 'Close' in dados:
                dados = dados[['Close']]
            elif 'Adj Close' in dados:
                dados = dados[['Adj Close']]
            else:
                st.warning(f"⚠️ Nenhuma coluna 'Close' ou 'Adj Close' encontrada para {setor}")
                continue

            if dados.empty:
                st.warning(f"⚠️ Dados vazios para setor: {setor}")
                continue

            dados = dados.fillna(method='ffill')
            retornos = dados.pct_change().dropna()
            media_mensal = retornos.mean(axis=1)
            retornos_setoriais[setor] = media_mensal
        except Exception as e:
            st.error(f"⚠️ Erro ao processar setor {setor}: {e}")
            continue

    # Gerar dados simulados para setores sem dados
    setores_sem_dados = [setor for setor in setores.keys() if setor not in retornos_setoriais]
    if setores_sem_dados:
        for setor in setores_sem_dados:
            tickers_no_setor = setores[setor]
            simulados = gerar_dados_simulados(tickers_no_setor)
            retornos_setoriais[setor] = simulados.mean(axis=1)

    retornos_df = pd.DataFrame(retornos_setoriais).dropna()
    st.info(f"📈 Retornos setoriais disponíveis: {list(retornos_df.columns)}")

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
            st.success(f"✅ Regressão bem-sucedida para setor {setor}")
        except Exception as e:
            st.error(f"⚠️ Regressão falhou para setor {setor}: {e}")
            continue

    if not coeficientes:
        st.error("⚠️ Nenhum coeficiente foi gerado. Retornando dicionário vazio.")
        return {}

    st.info(f"📈 Coeficientes finais: {coeficientes}")

    if normalizar:
        coeficientes = normalizar_coeficientes(coeficientes)

    if salvar_csv:
        try:
            pd.DataFrame.from_dict(coeficientes, orient='index').to_csv("sensibilidade_setorial.csv")
            st.success("✅ Coeficientes salvos em sensibilidade_setorial.csv")
        except Exception as e:
            st.error(f"⚠️ Falha ao salvar coeficientes em CSV: {e}")

    return coeficientes


def normalizar_coeficientes(coef_dict):
    """Normaliza os coeficientes para uma escala fixa."""
    return {
        setor: {
            fator: int(np.clip(round(valor * 2), -2, 2)) for fator, valor in coef.items()
        } for setor, coef in coef_dict.items()
    }

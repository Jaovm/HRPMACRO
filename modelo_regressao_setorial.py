import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from collections import defaultdict
from dados_setoriais import setores_por_ticker
from time import sleep


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


def baixar_dados_com_retentativa(tickers, period="2y", interval="1mo", max_retentativas=3):
    """Baixa dados com retentativas em caso de falha e valida colunas."""
    for tentativa in range(max_retentativas):
        try:
            sleep(1)  # Aguarda 1 segundo entre as tentativas para evitar limitações
            dados = yf.download(tickers, period=period, interval=interval, group_by="ticker", auto_adjust=False)
            
            # Logando os dados retornados
            st.write("🔍 Dados brutos retornados pelo yfinance:")
            st.write(dados)

            # Tentar acessar a coluna 'Adj Close' de forma flexível
            if isinstance(dados.columns, pd.MultiIndex):
                if 'Adj Close' in dados.columns.get_level_values(0):
                    st.info("✅ Coluna 'Adj Close' encontrada no MultiIndex.")
                    return dados['Adj Close']
                
                # Verificando a 5ª posição
                st.warning("⚠️ Coluna 'Adj Close' não encontrada pelo nome. Tentando acessar pela posição.")
                return dados.iloc[:, 4]  # Acessar a 5ª coluna diretamente

            elif 'Adj Close' in dados.columns:
                st.info("✅ Coluna 'Adj Close' encontrada.")
                return dados['Adj Close']
            
            # Verificando a 5ª posição diretamente
            st.warning("⚠️ Coluna 'Adj Close' não encontrada. Tentando acessar pela posição.")
            return dados.iloc[:, 4]  # Acessar a 5ª coluna diretamente

        except Exception as e:
            st.warning(f"⚠️ Tentativa {tentativa + 1}/{max_retentativas} falhou para tickers {tickers}. ({e})")
            sleep(5)  # Espera antes de tentar novamente

    st.error(f"⚠️ Não foi possível baixar os dados para tickers {tickers} após {max_retentativas} tentativas.")
    return pd.DataFrame()  # Retorna DataFrame vazio em caso de falha


def gerar_dados_simulados(tickers, periodos=24):
    """Gera dados simulados para tickers sem dados válidos."""
    st.warning("⚠️ Gerando dados simulados para os tickers ausentes...")
    datas = pd.date_range(end=pd.Timestamp.today(), periods=periodos, freq='ME')
    dados_simulados = {ticker: np.random.normal(0, 0.02, len(datas)) for ticker in tickers}
    return pd.DataFrame(dados_simulados, index=datas)


def integrar_dados_simulados(setores, tickers_sem_dados):
    """Integra dados simulados para setores que falharam ao obter dados reais."""
    retornos_setoriais = {}
    for setor, tickers in setores.items():
        if any(ticker in tickers_sem_dados for ticker in tickers):
            st.warning(f"⚠️ Usando dados simulados para o setor {setor}")
            simulados = gerar_dados_simulados(tickers)
            retornos_setoriais[setor] = simulados.mean(axis=1)
    return retornos_setoriais


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
    tickers_sem_dados = set([t for setor in setores.values() for t in setor]) - set(tickers_validos)
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

            if dados.empty:
                st.warning(f"⚠️ Dados vazios para setor: {setor}")
                continue

            dados = dados.fillna(method='ffill')
            retornos = dados.pct_change().dropna()
            if retornos.empty:
                st.warning(f"⚠️ Retornos vazios para setor: {setor}")
                continue

            retornos_setoriais[setor] = retornos.mean(axis=1)
        except Exception as e:
            st.error(f"⚠️ Erro ao processar setor {setor}: {e}")
            continue

    # Integração de dados simulados
    dados_simulados = integrar_dados_simulados(setores, tickers_sem_dados)
    retornos_setoriais.update(dados_simulados)

    retornos_df = pd.DataFrame(retornos_setoriais).dropna()
    st.info(f"📈 Retornos setoriais disponíveis: {list(retornos_df.columns)}")

    if retornos_df.empty:
        st.error("⚠️ Nenhum dado de retorno setorial disponível após filtragem.")
        return {}

    st.info("✅ Fluxo concluído com sucesso!")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

# ========= DICIONÁRIOS ==========

# Ativos e pesos atuais
carteira_atual = {
    'AGRO3.SA': 10.0,
    'BBAS3.SA': 1.2,
    'BBSE3.SA': 6.5,
    'BPAC11.SA': 10.6,
    'EGIE3.SA': 5.0,
    'ITUB3.SA': 0.5,
    'PRIO3.SA': 15.0,
    'PSSA3.SA': 15.0,
    'SAPR3.SA': 6.7,
    'SBSP3.SA': 4.0,
    'VIVT3.SA': 6.4,
    'WEGE3.SA': 15.0,
    'TOTS3.SA': 1.0,
    'B3SA3.SA': 0.1,
    'TAEE3.SA': 3.0
}

# Define setores por ativo
setores_por_ticker = {
    'AGRO3.SA': 'Consumo básico',
    'BBAS3.SA': 'Financeiro',
    'BBSE3.SA': 'Financeiro',
    'BPAC11.SA': 'Financeiro',
    'EGIE3.SA': 'Utilidades',
    'ITUB3.SA': 'Financeiro',
    'PRIO3.SA': 'Energia',
    'PSSA3.SA': 'Financeiro',
    'SAPR3.SA': 'Utilidades',
    'SBSP3.SA': 'Utilidades',
    'VIVT3.SA': 'Comunicações',
    'WEGE3.SA': 'Indústria',
    'TOTS3.SA': 'Tecnologia',
    'B3SA3.SA': 'Financeiro',
    'TAEE3.SA': 'Utilidades'
}

setores_por_cenario = {
    "Expansionista": ['Consumo discricionário', 'Tecnologia', 'Indústria'],
    "Neutro": ['Saúde', 'Financeiro', 'Utilidades', 'Varejo'],
    "Restritivo": ['Utilidades', 'Energia', 'Saúde', 'Consumo básico']
}

# ========= FUNÇÕES ==========

def get_bcb(code):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados/ultimos/1?formato=json"
    r = requests.get(url)
    return float(r.json()[0]['valor'].replace(",", ".")) if r.status_code == 200 else None

def obter_macro():
    return {
        "selic": get_bcb(432),
        "ipca": get_bcb(433),
        "dolar": get_bcb(1)
    }

def classificar_cenario_macro(m):
    if m['ipca'] > 5 or m['selic'] > 12:
        return "Restritivo"
    elif m['ipca'] < 4 and m['selic'] < 10:
        return "Expansionista"
    else:
        return "Neutro"

def obter_preco_alvo(ticker):
    try:
        return yf.Ticker(ticker).info.get('targetMeanPrice', None)
    except:
        return None

def obter_preco_atual(ticker):
    try:
        return yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    except:
        return None

# ========= FILTRAR AÇÕES ==========

def filtrar_ativos_validos(carteira, cenario):
    setores_bons = setores_por_cenario[cenario]
    ativos_validos = []

    for ticker in carteira:
        setor = setores_por_ticker.get(ticker, None)
        preco_atual = obter_preco_atual(ticker)
        preco_alvo = obter_preco_alvo(ticker)

        if preco_atual is None or preco_alvo is None:
            continue
        if preco_atual < preco_alvo:
            ativos_validos.append({
                "ticker": ticker,
                "setor": setor,
                "preco_atual": preco_atual,
                "preco_alvo": preco_alvo,
                "favorecido": setor in setores_bons
            })

    return ativos_validos

# ========= OTIMIZAÇÃO CORRIGIDA ==========

def obter_preco_diario_ajustado(tickers):
    dados_brutos = yf.download(tickers, period="3y", auto_adjust=False)

    if isinstance(dados_brutos.columns, pd.MultiIndex):
        # Vários ativos — usa MultiIndex com 'Adj Close'
        if 'Adj Close' in dados_brutos.columns.get_level_values(0):
            return dados_brutos['Adj Close']
        elif 'Close' in dados_brutos.columns.get_level_values(0):
            return dados_brutos['Close']
        else:
            raise ValueError("Colunas 'Adj Close' ou 'Close' não encontradas nos dados.")
    else:
        # Apenas 1 ativo — dados_brutos tem colunas simples
        if 'Adj Close' in dados_brutos.columns:
            return dados_brutos[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
        elif 'Close' in dados_brutos.columns:
            return dados_brutos[['Close']].rename(columns={'Close': tickers[0]})
        else:
            raise ValueError("Coluna 'Adj Close' ou 'Close' não encontrada nos dados.")

def otimizar_carteira_sharpe(tickers, min_pct=0.01, max_pct=0.30):
    # Verifica se há dados ausentes ou inválidos nos retornos
    dados = obter_preco_diario_ajustado(tickers)
    retornos = dados.pct_change().dropna()

    # Verifica e limpa valores inválidos ou infinitos
    if retornos.isnull().any().any() or np.isinf(retornos.values).any():
        st.warning("Os dados de retornos contêm valores inválidos ou ausentes. Verifique a qualidade dos dados.")
        return None

    # Calcula a média anualizada e a matriz de covariância com Ledoit-Wolf
    medias = retornos.mean() * 252
    cov = LedoitWolf().fit(retornos).covariance_

    n = len(tickers)

    # Função de objetivo para maximizar o Sharpe
    def sharpe_neg(pesos):
        retorno_esperado = np.dot(pesos, medias)
        volatilidade = np.sqrt(np.dot(pesos.T, np.dot(cov, pesos)))
        return -retorno_esperado / volatilidade

    # Inicializa os pesos dentro das restrições e com a soma igual a 1
    init = np.array([1/n] * n)

    # Garantir que os pesos estão dentro dos limites e somam 1
    def verificar_restricoes(pesos):
        if np.any(pesos < min_pct) or np.any(pesos > max_pct):
            return np.inf  # Penaliza soluções que violam restrições
        return np.sum(pesos) - 1  # Soma deve ser 1

    # Restrição para garantir que a soma dos pesos seja 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # Restrições de alocação mínima e máxima por ativo
    bounds = tuple((min_pct, max_pct) for _ in range(n))

    # Tentando otimizar com uma abordagem mais robusta
    try:
        resultado = minimize(sharpe_neg, init, bounds=bounds, constraints=constraints, method='trust-constr')
        if resultado.success:
            return resultado.x
        else:
            st.error(f"Otimização falhou: {resultado.message}")
            return None
    except Exception as e:
        st.error(f"Erro na otimização: {str(e)}")
        return None

# ========= STREAMLIT ==========

st.set_page_config(page_title="Sugestão de Carteira", layout="wide")
st.title("📊 Sugestão e Otimização de Carteira com Base no Cenário Macroeconômico")

# MACRO
macro = obter_macro()
cenario = classificar_cenario_macro(macro)
col1, col2, col3 = st.columns(3)
col1.metric("Selic (%)", f"{macro['selic']:.2f}")
col2.metric("Inflação IPCA (%)", f"{macro['ipca']:.2f}")
col3.metric("Dólar (R$)", f"{macro['dolar']:.2f}")
st.info(f"**Cenário Macroeconômico Atual:** {cenario}")

# INPUT
st.subheader("📌 Informe sua carteira atual")
tickers = st.text_input("Tickers separados por vírgula", ", ".join(carteira_atual.keys())).upper()
carteira = [t.strip() for t in tickers.split(",") if t.strip()]

aporte_mensal = st.number_input("Valor do aporte mensal (R$)", min_value=0, value=500)

if st.button("Gerar Alocação Otimizada e Aporte"):
    ativos_validos = filtrar_ativos_validos(carteira, cenario)

    if not ativos_validos:
        st.warning("Nenhum ativo com preço atual abaixo do preço-alvo dos analistas.")
    else:
        tickers_validos = [a['ticker'] for a in ativos_validos]
        try:
            pesos = otimizar_carteira_sharpe(tickers_validos)
            if pesos is not None:
                # Calcula a nova alocação considerando o aporte
                aporte_total = aporte_mensal
                aporte_distribuido = pesos * aporte_total
                
                # Atualiza a tabela com os pesos atuais, novos e o aporte
                df_resultado = pd.DataFrame(ativos_validos)
                df_resultado["Alocação Atual (%)"] = df_resultado["ticker"].map(carteira_atual)
                df_resultado["Alocação Nova (%)"] = (pesos * 100).round(2)
                df_resultado["Aporte (R$)"] = (aporte_distribuido).round(2)
                df_resultado = df_resultado.sort_values("Alocação Nova (%)", ascending=False)
                
                st.success("✅ Carteira otimizada com Sharpe máximo (restrições relaxadas: 1%-30%).")
                st.dataframe(df_resultado[["ticker", "setor", "preco_atual", "preco_alvo", "Alocação Atual (%)", "Alocação Nova (%)", "Aporte (R$)"]])
            else:
                st.error("Falha na otimização da carteira.")
        except Exception as e:
            st.error(f"Erro na otimização: {str(e)}")

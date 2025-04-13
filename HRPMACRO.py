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
    'AGRO3.SA': 'Agro',
    'BBAS3.SA': 'Banco',
    'BBSE3.SA': 'Seguradora',
    'BPAC11.SA': 'Banco',
    'EGIE3.SA': 'Energia',
    'ITUB3.SA': 'Banco',
    'PRIO3.SA': 'Petróleo',
    'PSSA3.SA': 'Seguradora',
    'SAPR3.SA': 'Saneamento',
    'SBSP3.SA': 'Saneamento',
    'VIVT3.SA': 'Telecomunicação',
    'WEGE3.SA': 'Indústria',
    'TOTS3.SA': 'Tecnologia',
    'B3SA3.SA': 'Bolsa',
    'TAEE3.SA': 'Energia',
}

setores_por_cenario = {
    "Expansionista": ['Tecnologia', 'Indústria', 'Agro', 'Consumo discricionário'],
    "Neutro": ['Banco', 'Seguradora', 'Telecomunicação', 'Saneamento'],
    "Restritivo": ['Energia', 'Petróleo', 'Saneamento', 'Consumo básico']
}

# ========= FUNÇÕES AUXILIARES ==========

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
        if 'Adj Close' in dados_brutos.columns.get_level_values(0):
            return dados_brutos['Adj Close']
        elif 'Close' in dados_brutos.columns.get_level_values(0):
            return dados_brutos['Close']
        else:
            raise ValueError("Colunas 'Adj Close' ou 'Close' não encontradas nos dados.")
    else:
        if 'Adj Close' in dados_brutos.columns:
            return dados_brutos[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
        elif 'Close' in dados_brutos.columns:
            return dados_brutos[['Close']].rename(columns={'Close': tickers[0]})
        else:
            raise ValueError("Coluna 'Adj Close' ou 'Close' não encontrada nos dados.")

def otimizar_carteira_sharpe(tickers, min_pct=0.01, max_pct=0.30, pesos_setor=None):
    dados = obter_preco_diario_ajustado(tickers)
    retornos = dados.pct_change().dropna()

    if retornos.isnull().any().any() or np.isinf(retornos.values).any():
        st.warning("Os dados de retornos contêm valores inválidos ou ausentes. Verifique a qualidade dos dados.")
        return None

    medias = retornos.mean() * 252
    cov = LedoitWolf().fit(retornos).covariance_

    n = len(tickers)

    def sharpe_neg(pesos):
        pesos_ajustados = np.array([peso * pesos_setor[setores_por_ticker.get(ticker, '')] if setores_por_ticker.get(ticker, '') in pesos_setor else peso for ticker, peso in zip(tickers, pesos)])
        
        retorno_esperado = np.dot(pesos_ajustados, medias)
        volatilidade = np.sqrt(np.dot(pesos_ajustados.T, np.dot(cov, pesos_ajustados)))
        return -retorno_esperado / volatilidade

    init = np.array([1/n] * n)

    def verificar_restricoes(pesos):
        if np.any(pesos < min_pct) or np.any(pesos > max_pct):
            return np.inf  
        return np.sum(pesos) - 1

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((min_pct, max_pct) for _ in range(n))

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

        # Peso de cada setor baseado no cenário macroeconômico
        pesos_setor = {setor: 1 for setor in setores_por_cenario[cenario]}

        try:
            pesos = otimizar_carteira_sharpe(tickers_validos, pesos_setor=pesos_setor)
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

                # Sugestões de compra
                st.subheader("💡 Sugestões de Compra")
                for ativo in ativos_validos:
                    if ativo['preco_atual'] < ativo['preco_alvo']:
                        st.write(f"**{ativo['ticker']}** - Setor: {ativo['setor']} | Preço Atual: R$ {ativo['preco_atual']} | Preço Alvo: R$ {ativo['preco_alvo']} (Comprar!)")
            else:
                st.error("Falha na otimização da carteira.")
        except Exception as e:
            st.error(f"Erro na otimização: {str(e)}")

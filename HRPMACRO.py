import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

# ========= DICIONÁRIOS ==========

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

def obter_preco_diario_ajustado(tickers):
    dados_brutos = yf.download(tickers, period="3y", auto_adjust=False)
    if isinstance(dados_brutos.columns, pd.MultiIndex):
        if 'Adj Close' in dados_brutos.columns.get_level_values(0):
            return dados_brutos['Adj Close']
        elif 'Close' in dados_brutos.columns.get_level_values(0):
            return dados_brutos['Close']
    else:
        if 'Adj Close' in dados_brutos.columns:
            return dados_brutos[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
        elif 'Close' in dados_brutos.columns:
            return dados_brutos[['Close']].rename(columns={'Close': tickers[0]})
    raise ValueError("Coluna 'Adj Close' ou 'Close' não encontrada nos dados.")

def otimizar_carteira_sharpe(tickers, min_pct=0.01, max_pct=0.30, pesos_setor=None):
    dados = obter_preco_diario_ajustado(tickers)
    retornos = dados.pct_change().dropna()
    medias = retornos.mean() * 252
    cov = LedoitWolf().fit(retornos).covariance_
    n = len(tickers)

    def sharpe_neg(pesos):
        pesos_ajustados = np.array([
            peso * pesos_setor[setores_por_ticker.get(t, '')] if setores_por_ticker.get(t, '') in pesos_setor else peso
            for t, peso in zip(tickers, pesos)
        ])
        retorno = np.dot(pesos_ajustados, medias)
        vol = np.sqrt(np.dot(pesos_ajustados.T, np.dot(cov, pesos_ajustados)))
        return -retorno / vol

    init = np.array([1/n] * n)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((min_pct, max_pct) for _ in range(n))

    resultado = minimize(sharpe_neg, init, bounds=bounds, constraints=constraints, method='trust-constr')
    return resultado.x if resultado.success else None

# ========= STREAMLIT ==========

st.set_page_config(page_title="Alocação Após Aporte", layout="wide")
st.title("📊 Alocação Após Aporte com Base no Cenário Macroeconômico")

macro = obter_macro()
cenario = classificar_cenario_macro(macro)
col1, col2, col3 = st.columns(3)
col1.metric("Selic (%)", f"{macro['selic']:.2f}")
col2.metric("IPCA (%)", f"{macro['ipca']:.2f}")
col3.metric("Dólar (R$)", f"{macro['dolar']:.2f}")
st.info(f"**Cenário Atual:** {cenario}")

tickers = list(carteira_atual.keys())
aporte_mensal = st.number_input("Aporte mensal (R$)", min_value=0, value=500)

if st.button("Gerar Alocação com Aporte e Sugestões"):
    ativos_validos = filtrar_ativos_validos(carteira, cenario)

    if not ativos_validos:
        st.warning("Nenhum ativo com preço atual abaixo do preço-alvo dos analistas.")
    else:
        tickers_validos = [a['ticker'] for a in ativos_validos]

        # Peso de cada setor baseado no cenário macroeconômico
        pesos_setor = {setor: 1.2 if setor in setores_por_cenario[cenario] else 1 for setor in set(setores_por_ticker.values())}

        try:
            pesos = otimizar_carteira_sharpe(tickers_validos, pesos_setor=pesos_setor)
            if pesos is not None:
                df = pd.DataFrame(ativos_validos)
                df["Alocação Atual (%)"] = df["ticker"].map(carteira_atual)

                # Cálculos em R$ (assumindo carteira atual de R$1000)
                df["Valor Atual (R$)"] = df["Alocação Atual (%)"] / 100 * 1000
                df["Aporte (R$)"] = (pesos * aporte_mensal).round(2)
                df["Valor Total (R$)"] = df["Valor Atual (R$)"] + df["Aporte (R$)"]
                df["Nova Alocação (%)"] = df["Valor Total (R$)"] / (1000 + aporte_mensal) * 100
                df["Favorecido no Cenário"] = df["setor"].apply(lambda s: "✅" if s in setores_por_cenario[cenario] else "")

                df = df.sort_values("Nova Alocação (%)", ascending=False)

                st.success("✅ Alocação após o aporte gerada com base no cenário macroeconômico.")
                st.dataframe(df[[
                    "ticker", "setor", "preco_atual", "preco_alvo",
                    "Alocação Atual (%)", "Nova Alocação (%)",
                    "Aporte (R$)", "Favorecido no Cenário"
                ]])

                # Sugestões de compra
                st.subheader("💡 Sugestões de Compra")
                for ativo in df.itertuples():
                    if ativo.preco_atual < ativo.preco_alvo:
                        st.write(f"**{ativo.ticker}** | Setor: {ativo.setor} | Preço Atual: R$ {ativo.preco_atual:.2f} | Alvo: R$ {ativo.preco_alvo:.2f} {'✅ Favorecido' if ativo.setor in setores_por_cenario[cenario] else ''}")
            else:
                st.error("Falha na otimização da carteira.")
        except Exception as e:
            st.error(f"Erro na otimização: {str(e)}")

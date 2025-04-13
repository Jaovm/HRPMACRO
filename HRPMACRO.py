import streamlit as st
import pandas as pd
import yfinance as yf
import requests

# ========== MAPAS AUXILIARES ==========
setores_por_ticker = {
    'WEGE3.SA': 'Indústria',
    'PETR4.SA': 'Energia',
    'VIVT3.SA': 'Utilidades',
    'EGIE3.SA': 'Utilidades',
    'ITUB4.SA': 'Financeiro',
    'LREN3.SA': 'Consumo discricionário',
    'ABEV3.SA': 'Consumo básico',
    'B3SA3.SA': 'Financeiro',
    'MGLU3.SA': 'Consumo discricionário',
    'HAPV3.SA': 'Saúde',
    'RADL3.SA': 'Saúde',
    'RENT3.SA': 'Consumo discricionário',
    'VALE3.SA': 'Indústria',
    'TOTS3.SA': 'Tecnologia',
    # adicione mais tickers conforme desejar
}

setores_por_cenario = {
    "Expansionista": ['Consumo discricionário', 'Tecnologia', 'Indústria'],
    "Neutro": ['Saúde', 'Financeiro', 'Utilidades', 'Varejo'],
    "Restritivo": ['Utilidades', 'Energia', 'Saúde', 'Consumo básico']
}

# ========== FUNÇÕES ==========
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

def sugerir_nova_alocacao(carteira, cenario):
    setores_bons = setores_por_cenario[cenario]
    alocacao = []

    for ticker in carteira:
        setor = setores_por_ticker.get(ticker, None)
        preco_atual = obter_preco_atual(ticker)
        preco_alvo = obter_preco_alvo(ticker)

        if preco_alvo is None or preco_atual is None or preco_atual >= preco_alvo:
            continue

        bonus = 1.5 if setor in setores_bons else 1.0
        potencial = (preco_alvo / preco_atual - 1) * 100 * bonus

        alocacao.append({
            "Ticker": ticker,
            "Setor": setor or "Desconhecido",
            "Preço Atual": round(preco_atual, 2),
            "Preço Alvo": round(preco_alvo, 2),
            "Potencial (%)": round(potencial, 2)
        })

    df = pd.DataFrame(alocacao).sort_values(by="Potencial (%)", ascending=False)
    if not df.empty:
        df["Nova Alocação (%)"] = round(df["Potencial (%)"] / df["Potencial (%)"].sum() * 100, 2)
    return df

# ========== INTERFACE ==========
st.title("🏦 Sugestão de Alocação Baseada no Cenário Macroeconômico")

macro = obter_macro()
cenario = classificar_cenario_macro(macro)

col1, col2, col3 = st.columns(3)
col1.metric("Selic (%)", f"{macro['selic']:.2f}")
col2.metric("Inflação IPCA (%)", f"{macro['ipca']:.2f}")
col3.metric("Dólar (R$)", f"{macro['dolar']:.2f}")
st.info(f"**Cenário Macroeconômico Atual:** {cenario}")

st.subheader("📌 Informe sua carteira")
tickers = st.text_input("Tickers separados por vírgula", "WEGE3.SA, PETR4.SA, VIVT3.SA, TOTS3.SA").upper()
carteira = [t.strip() for t in tickers.split(",") if t.strip()]

if st.button("Gerar Nova Alocação"):
    df = sugerir_nova_alocacao(carteira, cenario)
    if df.empty:
        st.warning("Nenhum ativo com preço atual abaixo do preço-alvo.")
    else:
        st.success("Sugestão de nova alocação com base em cenário e preço-alvo:")
        st.dataframe(df)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
import plotly.express as px

# ========= DICIONÁRIOS ==========

setores_por_ticker = {
    'WEGE3.SA': 'Indústria', 'PETR4.SA': 'Energia', 'VIVT3.SA': 'Utilidades',
    'EGIE3.SA': 'Energia', 'ITUB4.SA': 'Financeiro', 'LREN3.SA': 'Consumo discricionário',
    'ABEV3.SA': 'Consumo básico', 'B3SA3.SA': 'Financeiro', 'MGLU3.SA': 'Consumo discricionário',
    'HAPV3.SA': 'Saúde', 'RADL3.SA': 'Saúde', 'RENT3.SA': 'Consumo discricionário',
    'VALE3.SA': 'Indústria', 'TOTS3.SA': 'Tecnologia', 'AGRO3.SA': 'Agronegócio',
    'BBAS3.SA': 'Financeiro', 'BBSE3.SA': 'Seguradoras', 'BPAC11.SA': 'Financeiro',
    'PRIO3.SA': 'Petróleo', 'PSSA3.SA': 'Seguradoras', 'SAPR3.SA': 'Utilidades',
    'SBSP3.SA': 'Utilidades', 'TAEE3.SA': 'Energia'
}

setores_por_cenario = {
    "Expansionista": ['Consumo discricionário', 'Tecnologia', 'Indústria', 'Agronegócio'],
    "Neutro": ['Saúde', 'Financeiro', 'Utilidades', 'Varejo', 'Seguradoras'],
    "Restritivo": ['Utilidades', 'Energia', 'Saúde', 'Consumo básico', 'Petróleo']
}

empresas_exportadoras = ['AGRO3.SA', 'PRIO3.SA']

explicacoes_ativos = {
    "PRIO3.SA": "Setor de petróleo, favorecido no cenário restritivo. Exportadora beneficiada pelo dólar alto e petróleo acima de US$80, o que deu bônus duplo no score. Historicamente teve bom retorno ajustado ao risco.",
    "AGRO3.SA": "Exportadora agrícola, com bônus do dólar alto. Favorecida no cenário expansionista, mas o alto upside compensa o cenário atual. Bom Sharpe histórico.",
    "WEGE3.SA": "Industrial que se beneficia de crescimento consistente e baixa volatilidade. Mesmo não sendo favorecida no cenário atual, tem Sharpe elevado.",
    "SBSP3.SA": "Utilidade pública favorecida no cenário restritivo. Pode ter estabilidade e algum upside.",
    "VIVT3.SA": "Do setor de utilidades, se beneficia de cenário restritivo. Estável e com bom histórico.",
    "BBAS3.SA": "Banco com estabilidade e presença no setor financeiro, neutro no cenário atual.",
    "BPAC11.SA": "Instituição financeira que pode ter alta volatilidade, mas bom retorno recente.",
    "PSSA3.SA": "Seguradora com perfil defensivo, favorecida em cenários neutros e restritivos.",
    "B3SA3.SA": "Do setor financeiro. Alta correlação com mercado, mas com bom desempenho histórico.",
    "EGIE3.SA": "Empresa de energia elétrica. Estável, favorecida no cenário restritivo, mas retorno mais moderado.",
    "BBSE3.SA": "Seguradora com perfil defensivo e retorno mais estável.",
    "TOTS3.SA": "Tecnologia. Favorecida em cenário expansionista, mas no atual pode apresentar maior risco com menor retorno ajustado."
}

# ========= MACRO ==========
def get_bcb(code):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados/ultimos/1?formato=json"
    r = requests.get(url)
    return float(r.json()[0]['valor'].replace(",", ".")) if r.status_code == 200 else None

def obter_preco_petroleo():
    try:
        return float(yf.Ticker("CL=F").history(period="1d")['Close'].iloc[-1])
    except:
        return None

def obter_macro():
    return {
        "selic": get_bcb(432),
        "ipca": get_bcb(433),
        "dolar": get_bcb(1),
        "petroleo": obter_preco_petroleo()
    }

def classificar_cenario_macro(m):
    if m['ipca'] > 5 or m['selic'] > 12:
        return "Restritivo"
    elif m['ipca'] < 4 and m['selic'] < 10:
        return "Expansionista"
    else:
        return "Neutro"

# ========= PREÇO ALVO ==========
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
def calcular_score(preco_atual, preco_alvo, favorecido, ticker, macro):
    if preco_atual is None or preco_alvo is None or preco_atual == 0:
        return -np.inf  # Garante que esse ativo nunca será escolhido

    upside = (preco_alvo - preco_atual) / preco_atual
    bonus = 0.1 if favorecido else 0
    if ticker in empresas_exportadoras:
        if macro['dolar'] and macro['dolar'] > 5:
            bonus += 0.05
        if macro['petroleo'] and macro['petroleo'] > 80:
            bonus += 0.05
    return upside + bonus


def filtrar_ativos_validos(carteira, cenario, macro):
    setores_bons = setores_por_cenario[cenario]
    ativos_validos = []

    for ticker in carteira:
        setor = setores_por_ticker.get(ticker, None)
        preco_atual = obter_preco_atual(ticker)
        preco_alvo = obter_preco_alvo(ticker)

        if preco_atual is None or preco_alvo is None:
            continue
        if preco_atual < preco_alvo:
            favorecido = setor in setores_bons
            score = calcular_score(preco_atual, preco_alvo, favorecido, ticker, macro)
            ativos_validos.append({
                "ticker": ticker,
                "setor": setor,
                "preco_atual": preco_atual,
                "preco_alvo": preco_alvo,
                "favorecido": favorecido,
                "score": score,
                "explicacao": explicacoes_ativos.get(ticker, "")
            })

    ativos_validos.sort(key=lambda x: x['score'], reverse=True)
    return ativos_validos

def otimizar_carteira_sharpe(tickers):
    dados = obter_preco_diario_ajustado(tickers)
    retornos = dados.pct_change().dropna()
    media_retorno = retornos.mean()
    cov = LedoitWolf().fit(retornos).covariance_

    def sharpe_negativo(pesos):
        retorno = np.dot(pesos, media_retorno)
        volatilidade = np.sqrt(np.dot(pesos.T, np.dot(cov, pesos)))
        return -retorno / volatilidade if volatilidade != 0 else 0

    n = len(tickers)
    restricoes = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    limites = tuple((0, 1) for _ in range(n))
    pesos_iniciais = np.array([1. / n] * n)

    resultado = minimize(sharpe_negativo, pesos_iniciais, method='SLSQP', bounds=limites, constraints=restricoes)

    if resultado.success:
        return resultado.x
    else:
        raise ValueError("Falha ao otimizar a carteira com base no Sharpe")

# ========= GRÁFICO DE ALOCAÇÃO ==========
def exibir_grafico_alocacao(df_resultado):
    fig = px.pie(
        df_resultado,
        names='ticker',
        values='Alocação (%)',
        title='Distribuição da Alocação por Ativo',
        hole=0.4
    )
    fig.update_traces(textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

# ========= INTERFACE STREAMLIT ==========
st.set_page_config(page_title="Alocação de Carteira Macro", layout="wide")
st.title("Análise e Alocação de Carteira com Base no Cenário Macroeconômico")

carteira = st.multiselect("Selecione os ativos da carteira:", list(setores_por_ticker.keys()), default=list(setores_por_ticker.keys()))
if st.button("Analisar carteira"):
    with st.spinner("Obtendo dados macroeconômicos e de mercado..."):
        macro = obter_macro()
        cenario = classificar_cenario_macro(macro)
        st.subheader("📌 Informe sua carteira atual")
default_carteira = "AGRO3.SA, BBAS3.SA, BBSE3.SA, BPAC11.SA, EGIE3.SA, ITUB3.SA, PRIO3.SA, PSSA3.SA, SAPR3.SA, SBSP3.SA, VIVT3.SA, WEGE3.SA, TOTS3.SA, B3SA3.SA, TAEE3.SA"
tickers = st.text_input("Tickers separados por vírgula", default_carteira).upper()
carteira = [t.strip() for t in tickers.split(",") if t.strip()]

aporte = st.number_input("💰 Valor do aporte mensal (R$)", min_value=100.0, value=1000.0, step=100.0)
usar_hrp = st.checkbox("Utilizar HRP em vez de Sharpe máximo")

if st.button("Gerar Alocação Otimizada"):
    ativos_validos = filtrar_ativos_validos(carteira, cenario, macro)

    if not ativos_validos:
        st.warning("Nenhum ativo com preço atual abaixo do preço-alvo dos analistas.")
    else:
        tickers_validos = [a['ticker'] for a in ativos_validos]
        try:
            if usar_hrp:
                pesos = otimizar_carteira_hrp(tickers_validos)
            else:
                pesos = otimizar_carteira_sharpe(tickers_validos)

            if pesos is not None:
                df_resultado = pd.DataFrame(ativos_validos)
                df_resultado["Alocação (%)"] = (pesos * 100).round(2)
                df_resultado["Valor Alocado (R$)"] = (pesos * aporte).round(2)
                df_resultado = df_resultado.sort_values("Alocação (%)", ascending=False)

                st.success("✅ Carteira otimizada com sucesso!")
                st.dataframe(df_resultado[["ticker", "setor", "preco_atual", "preco_alvo", "score", "Alocação (%)", "Valor Alocado (R$)"]])
            else:
                st.error("Falha na otimização da carteira.")
        except Exception as e:
            st.error(f"Erro na otimização: {str(e)}")

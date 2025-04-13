import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import requests
import datetime

def get_selic():
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4189/dados/ultimos/1?formato=json"
    r = requests.get(url).json()
    return float(r[0]['valor'].replace(',', '.'))

def get_ipca():
    url = "https://servicodados.ibge.gov.br/api/v3/agregados/433/dados/ultimos/1"
    r = requests.get(url).json()
    return float(r[0]['resultados'][0]['series'][0]['serie'].values()[0])

def get_pib():
    url = "https://servicodados.ibge.gov.br/api/v3/agregados/5932/periodos/ultimo/variaveis/4099?localidades=N1[all]"
    r = requests.get(url).json()
    valor = list(r[0]['resultados'][0]['series'][0]['serie'].values())[0]
    return float(valor)

def get_cambio():
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados/ultimos/1?formato=json"
    r = requests.get(url).json()
    return float(r[0]['valor'].replace(',', '.'))

def get_petroleo():
    hoje = datetime.datetime.today().strftime('%Y-%m-%d')
    petroleo = yf.download('BZ=F', end=hoje, period='5d')  # Brent
    return round(petroleo['Close'][-1], 2)

def classificar_cenario(selic, ipca, pib):
    if selic > 10 and ipca > 6 and pib < 0.5:
        return "Restritivo"
    elif selic < 7 and ipca < 4 and pib > 1.5:
        return "Expansionista"
    else:
        return "Neutro"

with st.expander("Cenário Macroeconômico Atual"):
    if st.button("Detectar Cenário Atual"):
        with st.spinner("Buscando dados macroeconômicos..."):
            try:
                selic = get_selic()
                ipca = get_ipca()
                pib = get_pib()
                cambio = get_cambio()
                petroleo = get_petroleo()
                cenario = classificar_cenario(selic, ipca, pib)

                st.success(f"Cenário atual: **{cenario}**")
                st.write(f"- SELIC: {selic:.2f}%")
                st.write(f"- IPCA: {ipca:.2f}%")
                st.write(f"- PIB: {pib:.2f}%")
                st.write(f"- Câmbio: R$ {cambio:.2f}")
                st.write(f"- Petróleo (Brent): US$ {petroleo}")

                st.session_state["cenario_atual"] = cenario
            except Exception as e:
                st.error("Erro ao buscar dados: " + str(e))

setores_por_cenario = {
    "Expansionista": ["Tecnologia", "Consumo", "Construção Civil", "Varejo", "Indústria"],
    "Restritivo": ["Energia", "Saúde", "Utilidades Públicas", "Bancos", "Seguradoras"],
    "Neutro": ["Telecom", "Exportadoras", "Serviços", "Bens de Capital"]
}

setores_ativos = {
    "AGRO3.SA": "Exportadoras",
    "BBAS3.SA": "Bancos",
    "BBSE3.SA": "Seguradoras",
    "BPAC11.SA": "Bancos",
    "EGIE3.SA": "Energia",
    "ITUB3.SA": "Bancos",
    "PRIO3.SA": "Petróleo",
    "PSSA3.SA": "Seguradoras",
    "SAPR3.SA": "Utilidades Públicas",
    "SBSP3.SA": "Utilidades Públicas",
    "VIVT3.SA": "Telecom",
    "WEGE3.SA": "Indústria",
    "TOTS3.SA": "Tecnologia",
    "B3SA3.SA": "Serviços",
    "TAEE3.SA": "Energia"
}

def recomendar_ativos_por_cenario(cenario, setores_ativos, setores_por_cenario):
    setores_favoraveis = setores_por_cenario.get(cenario, [])
    recomendados = [
        ativo for ativo, setor in setores_ativos.items()
        if setor in setores_favoraveis
    ]
    return recomendados, setores_favoraveis

def calcular_score(ativo, upside, setor, cenario, historico_bom=None):
    score = 0
    pesos = {
        "upside": 0.4,
        "setor_favoravel": 0.3,
        "historico": 0.2,
        "exportadora": 0.1
    }

    score += pesos["upside"] * upside.get(ativo, 0)
    if setor in setores_por_cenario.get(cenario, []):
        score += pesos["setor_favoravel"]
    if historico_bom and ativo in historico_bom:
        score += pesos["historico"]
    if setor == "Exportadoras" and cenario in ["Restritivo", "Neutro"]:  # dólar alto tende a favorecer exportadoras
        score += pesos["exportadora"]

    return round(score, 3)

# Definindo a lista de ativos
ativos = ['AGRO3.SA', 'BBAS3.SA', 'BBSE3.SA', 'BPAC11.SA', 'EGIE3.SA', 'ITUB3.SA', 'PRIO3.SA', 'PSSA3.SA', 'SAPR3.SA', 'SBSP3.SA', 'VIVT3.SA', 'WEGE3.SA', 'TOTS3.SA', 'B3SA3.SA', 'TAEE3.SA']

# Baixando os preços históricos dos ativos
inicio = '2017-01-01'
fim = '2025-01-01'
precos = yf.download(ativos, start=inicio, end=fim)['Adj Close']

# Calculando os retornos diários dos ativos
retornos = precos.pct_change().dropna()

# Calculando os retornos esperados e a matriz de covariância
retornos_esperados = retornos.mean() * 252  # anualizado
covariancia = retornos.cov() * 252  # anualizado

# Função para calcular o retorno do portfólio
def calc_retorno(pesos):
    return np.dot(pesos, retornos_esperados)

# Função para calcular o risco do portfólio
def calc_risco(pesos):
    return np.sqrt(np.dot(pesos.T, np.dot(covariancia, pesos)))

# Função objetivo (negativa do Sharpe ratio)
def objetivo(pesos):
    retorno_portfolio = calc_retorno(pesos)
    risco_portfolio = calc_risco(pesos)
    return -retorno_portfolio / risco_portfolio

# Restrições
restricao_soma = {'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1}  # soma dos pesos deve ser 1
restricoes = [restricao_soma]

# Limite de pesos individuais
limite = 0.05
limites = [(0, limite) for _ in range(len(ativos))]

# Função para otimizar o portfólio usando o método de minimização
resultado = minimize(objetivo, [1/len(ativos)]*len(ativos), method='SLSQP', bounds=limites, constraints=restricoes)
pesos_finais = dict(zip(ativos, resultado.x))

# Calculando o desempenho acumulado do portfólio
retorno_carteira = sum(retornos[ativo] * peso for ativo, peso in pesos_finais.items())
retorno_acumulado_carteira = (1 + retorno_carteira).cumprod()

# Baixando os dados do IBOV para comparação
precos_ibov = yf.download("^BVSP", start=inicio, end=fim)["Adj Close"]
retornos_ibov = precos_ibov.pct_change().dropna()
retorno_acumulado_ibov = (1 + retornos_ibov).cumprod()

# Criando o painel comparativo entre a carteira e o IBOV
df_comparativo = pd.DataFrame({
    "Carteira Otimizada": retorno_acumulado_carteira,
    "IBOV": retorno_acumulado_ibov
}).dropna()

# Calculando o score de cada ativo com base no retorno esperado, risco e setor
# (Aqui, consideramos uma fórmula simples de score: Retorno Esperado / Risco)
df_score = pd.DataFrame({
    "Ativo": ativos,
    "Score": [retornos_esperados[ativo] / np.sqrt(covariancia[ativo].sum()) for ativo in ativos]
})

# Exibindo os scores
st.subheader("🔍 Score de Ativos")
st.dataframe(df_score)

# Exibindo a alocação final
st.subheader("📊 Alocação Final do Portfólio")
df_alocacao = pd.DataFrame({
    "Ativo": list(pesos_finais.keys()),
    "Peso (%)": [round(p * 100, 2) for p in pesos_finais.values()]
})
st.dataframe(df_alocacao)

# Exportando a alocação para CSV
csv = df_alocacao.to_csv(index=False).encode('utf-8')
st.download_button("📥 Baixar Alocação (CSV)", data=csv, file_name="alocacao_portfolio.csv", mime='text/csv')

# Exibindo o gráfico comparativo entre a carteira otimizada e o IBOV
st.subheader("📊 Desempenho Carteira vs IBOV")
st.line_chart(df_comparativo)

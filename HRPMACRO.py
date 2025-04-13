import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

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

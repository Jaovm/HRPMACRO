import streamlit as st
import yfinance as yf
import numpy as np
import requests
import pandas as pd
from scipy.optimize import minimize

# Função para obter os dados da taxa SELIC
def get_selic():
    url = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=csv'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.text.splitlines()
        if len(data) > 1:
            selic_data = [line.split(';') for line in data]
            # Retorna a última taxa Selic, convertendo corretamente para float
            selic = selic_data[-1][1].replace('"', '').replace(',', '.')
            try:
                return float(selic)  # Convertendo para float para garantir que seja numérico
            except ValueError:
                st.error(f"Erro ao converter a Selic para número: {selic}")
                return None
        else:
            st.error("Nenhum dado encontrado na resposta da API da SELIC.")
            return None
    else:
        st.error(f"Erro ao acessar a API do Banco Central. Código de status: {response.status_code}")
        return None

# Função para calcular a inflação anual com base nos últimos 12 meses
def get_inflacao():
    url = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=csv'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.text.splitlines()
        if len(data) > 1:
            inflacao_data = [line.split(';') for line in data]
            
            # Pegando os últimos 12 meses de inflação mensal
            inflacao_mensal = [float(line[1].replace('"', '').replace(',', '.')) for line in inflacao_data[-12:]]
            
            # Calculando a inflação anual composta
            inflacao_anual = np.prod([1 + (x / 100) for x in inflacao_mensal]) - 1
            return round(inflacao_anual * 100, 2)  # Retorna a inflação anual em porcentagem
        else:
            st.error("Nenhum dado encontrado na resposta da API da inflação.")
            return None
    else:
        st.error(f"Erro ao acessar a API do Banco Central. Código de status: {response.status_code}")
        return None

# Função para sugerir alocação da carteira com base no cenário macroeconômico
def sugerir_alocacao_mercado(cenario_macroeconomico, pesos_atuais, ativos_selecionados):
    setores_favorecidos = cenario_macroeconomico['setores_favorecidos']
    alocacao_sugerida = pesos_atuais.copy()

    for setor in setores_favorecidos:
        # Filtrando ativos no setor favorecido
        ativos_favorecidos = [ativo for ativo, dados in ativos_selecionados.items() if dados['setor'] == setor]
        
        # Atualizando o peso de ativos nos setores favorecidos
        for ativo in ativos_favorecidos:
            alocacao_sugerida[ativo] = pesos_atuais.get(ativo, 0) + 0.05  # Adiciona um peso extra de 5% aos setores favorecidos

    # Normalizando os pesos para garantir que a soma seja 100%
    soma_pesos = sum(alocacao_sugerida.values())
    for ativo in alocacao_sugerida:
        alocacao_sugerida[ativo] = alocacao_sugerida[ativo] / soma_pesos

    return alocacao_sugerida

# Função principal para exibir o app
def main():
    st.title("Análise de Carteira de Investimentos e Cenário Macroeconômico")

    # Obter dados do cenário macroeconômico
    inflacao_anual = get_inflacao()
    selic = get_selic()
    meta_inflacao = 3.0  # Meta de inflação do Banco Central

    # Exibir cenário macroeconômico
    st.subheader("Cenário Macroeconômico Atual")
    st.write(f"Inflação Anual: {inflacao_anual}%")
    st.write(f"Taxa Selic: {selic}%")
    st.write(f"Meta de Inflação: {meta_inflacao}%")
    
    # Definindo ativos selecionados e seus pesos atuais
    ativos_selecionados = {
        "AGRO3.SA": {"setor": "Agronegócio", "peso_atual": 0.10},
        "BBAS3.SA": {"setor": "Financeiro", "peso_atual": 0.15},
        "BBSE3.SA": {"setor": "Seguradoras", "peso_atual": 0.12},
        "BPAC11.SA": {"setor": "Financeiro", "peso_atual": 0.20},
        "EGIE3.SA": {"setor": "Energia", "peso_atual": 0.10},
        "ITUB3.SA": {"setor": "Financeiro", "peso_atual": 0.10},
        "PRIO3.SA": {"setor": "Energia", "peso_atual": 0.10},
        "PSSA3.SA": {"setor": "Seguradoras", "peso_atual": 0.05},
        "SAPR3.SA": {"setor": "Utilities", "peso_atual": 0.05},
        "SBSP3.SA": {"setor": "Utilities", "peso_atual": 0.05},
        "VIVT3.SA": {"setor": "Telecomunicações", "peso_atual": 0.05},
        "WEGE3.SA": {"setor": "Indústria", "peso_atual": 0.10},
        "TOTS3.SA": {"setor": "Tecnologia", "peso_atual": 0.05},
        "B3SA3.SA": {"setor": "Financeiro", "peso_atual": 0.03},
        "TAEE3.SA": {"setor": "Energia", "peso_atual": 0.05},
    }

    # Simulando o cenário macroeconômico
    cenario_macroeconomico = {
        "setores_favorecidos": ["Energia", "Financeiro"]
    }

    # Sugerindo nova alocação
    alocacao_sugerida = sugerir_alocacao_mercado(cenario_macroeconomico, pesos_atuais={}, ativos_selecionados=ativos_selecionados)
    st.write("Nova alocação sugerida com base no cenário macroeconômico:")
    st.write(alocacao_sugerida)

if __name__ == "__main__":
    main()

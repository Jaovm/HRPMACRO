import requests
import streamlit as st

# Função para obter os dados da inflação anual
def get_inflacao():
    url = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=csv'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.text.splitlines()
        if len(data) > 1:
            # Processando os dados da inflação (assumindo que o índice 1 é o valor da inflação)
            inflacao_data = [line.split(';') for line in data]
            try:
                # A última inflação é a mais recente, então vamos pegar o último valor
                inflacao = inflacao_data[-1][1].replace('"', '').replace(',', '.')
                return round(float(inflacao), 2)  # Converte para float e arredonda para duas casas decimais
            except ValueError as e:
                st.error(f"Erro ao processar a inflação: {e}")
                return None
        else:
            st.error("Nenhum dado encontrado na resposta da API de Inflação.")
            return None
    else:
        st.error(f"Erro ao acessar a API do Banco Central. Código de status: {response.status_code}")
        return None

# Função para obter a taxa Selic
def get_selic():
    url = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.4186/dados?formato=csv'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.text.splitlines()
        if len(data) > 1:
            # Processando os dados da Selic (assumindo que o índice 1 é o valor da Selic)
            selic_data = [line.split(';') for line in data]
            try:
                # A última Selic é a mais recente, então vamos pegar o último valor
                selic = selic_data[-1][1].replace('"', '').replace(',', '.')
                return round(float(selic), 2)  # Converte para float e arredonda para duas casas decimais
            except ValueError as e:
                st.error(f"Erro ao processar a Selic: {e}")
                return None
        else:
            st.error("Nenhum dado encontrado na resposta da API de Selic.")
            return None
    else:
        st.error(f"Erro ao acessar a API do Banco Central. Código de status: {response.status_code}")
        return None

# Função principal para exibir o app
def main():
    st.title("Análise de Carteira de Investimentos e Cenário Macroeconômico")

    # Obter dados do cenário macroeconômico
    inflacao_anual = get_inflacao()  # Atualizado para corrigir erro de inflação
    selic = get_selic()  # Agora com a Selic corrigida
    meta_inflacao = 3.0  # Meta de inflação do Banco Central
    meta_selic = 13.75  # Meta de Selic (valor aproximado, é atualizado periodicamente pelo Banco Central)

    # Exibir cenário macroeconômico
    st.subheader("Cenário Macroeconômico Atual")
    if inflacao_anual:
        st.write(f"Inflação Anual: {inflacao_anual}%")
    if selic:
        st.write(f"Taxa Selic Atual: {selic}%")
    st.write(f"Meta de Inflação: {meta_inflacao}%")
    st.write(f"Meta de Selic: {selic}%")

    # Analisando o cenário econômico para definir uma tendência
    if inflacao_anual and inflacao_anual > meta_inflacao and selic >= 14.2:
        st.write("Cenário: Alta inflação e Selic alta, tendência de recessão ou inflação controlada.")
        cenario_macroeconomico = {
            "setores_favorecidos": ["Energia", "Financeiro"]
        }
    else:
        st.write("Cenário: Inflação controlada e Selic baixa, tendência de crescimento econômico.")
        cenario_macroeconomico = {
            "setores_favorecidos": ["Tecnologia", "Consumo", "Saúde"]
        }

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

    # Sugerindo nova alocação
    alocacao_sugerida = sugerir_alocacao_mercado(cenario_macroeconomico, pesos_atuais={}, ativos_selecionados=ativos_selecionados)
    
    st.write("Nova alocação sugerida com base no cenário macroeconômico:")
    for ativo, peso in alocacao_sugerida.items():
        st.write(f"{ativo}: {peso:.2%}")

if __name__ == "__main__":
    main()

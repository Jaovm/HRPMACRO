import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

# Função para obter dados financeiros
def obter_preco_diario_ajustado(tickers):
    df = yf.download(tickers, start="2018-01-01", end="2025-01-01")
    
    # Verificar todas as colunas disponíveis
    colunas_disponiveis = df.columns.tolist()
    
    # Preferir 'Adj Close', caso contrário, 'Close', caso contrário, qualquer coluna numérica
    if 'Adj Close' in colunas_disponiveis:
        return df['Adj Close']
    elif 'Close' in colunas_disponiveis:
        st.warning("A coluna 'Adj Close' não foi encontrada, utilizando 'Close'.")
        return df['Close']
    else:
        # Procurar por uma coluna numérica que provavelmente representa os preços
        for coluna in colunas_disponiveis:
            if pd.api.types.is_numeric_dtype(df[coluna]):
                st.warning(f"A coluna '{coluna}' foi usada como fallback para o preço.")
                return df[coluna]

    # Se nenhuma coluna adequada for encontrada, retornar erro
    st.error("Nenhuma coluna válida de preço ajustado ou fechado foi encontrada.")
    return pd.DataFrame()  # Retorna um DataFrame vazio em caso de erro

# Função para obter dados do Banco Central (BCB)
def get_bcb(code):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados/ultimos/1?formato=json"
    r = requests.get(url)
    return float(r.json()[0]['valor'].replace(",", ".")) if r.status_code == 200 else None

# Função para obter dados macroeconômicos
def obter_macro():
    return {
        "selic": get_bcb(432),
        "ipca": get_bcb(433),
        "dolar": get_bcb(1)
    }

# Função para classificar o cenário macroeconômico
def classificar_cenario_macro(m):
    if m['ipca'] > 5 or m['selic'] > 12:
        return "Restritivo"
    elif m['ipca'] < 4 and m['selic'] < 10:
        return "Expansionista"
    else:
        return "Neutro"

# Função de exibição do Streamlit
st.set_page_config(page_title="Sugestão de Carteira", layout="wide")
st.title("📊 Sugestão e Otimização de Carteira com Base no Cenário Macroeconômico")

# Exibindo os dados macroeconômicos
macro = obter_macro()
cenario = classificar_cenario_macro(macro)
col1, col2, col3 = st.columns(3)
col1.metric("Selic (%)", f"{macro['selic']:.2f}")
col2.metric("Inflação IPCA (%)", f"{macro['ipca']:.2f}")
col3.metric("Dólar (R$)", f"{macro['dolar']:.2f}")
st.info(f"**Cenário Macroeconômico Atual:** {cenario}")

# Entrada do usuário para a carteira
st.subheader("📌 Informe sua carteira atual")
tickers = st.text_input("Tickers separados por vírgula", "BBSE3.SA, ITUB3.SA").upper()
carteira = [t.strip() for t in tickers.split(",") if t.strip()]

aporte_mensal = st.number_input("Valor do aporte mensal (R$)", min_value=0, value=500)

if st.button("Gerar Alocação Otimizada e Aporte"):
    # Obtendo os preços ajustados de cada ativo
    dados = obter_preco_diario_ajustado(carteira)
    
    # Verificando se os dados de retorno são válidos
    if dados.empty:
        st.warning("Os dados de preços estão vazios. Verifique os tickers ou o período.")
    else:
        # Desempacotar o multiíndice, se necessário
        if isinstance(dados.columns, pd.MultiIndex):
            dados = dados.xs('Close', axis=1, level=0)  # Obtendo preços de fechamento
        
        # Calculando os retornos diários
        retornos = dados.pct_change().dropna()

        # Verificando se existem valores nulos ou infinitos nos retornos
        if retornos.isnull().any().any() or np.isinf(retornos.values).any():
            st.warning("Os dados de retornos contêm valores inválidos ou ausentes. Verifique a qualidade dos dados.")
        else:
            # Exemplo de exibição de retorno
            st.success("Dados de retorno calculados com sucesso.")
            st.dataframe(retornos.head())

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

# Função para obter preços ajustados ou de fechamento
def obter_preco_diario_ajustado(tickers):
    try:
        # Baixar os dados
        df = yf.download(tickers, start="2018-01-01", end="2025-01-01")
        
        # Verifica se há dados e se o dataframe tem colunas
        if df.empty:
            st.error("Os dados estão vazios para os tickers fornecidos.")
            return pd.DataFrame()
        
        # Se a coluna 'Adj Close' existe, usa ela. Caso contrário, usa 'Close'
        if 'Adj Close' in df.columns:
            return df['Adj Close']
        elif 'Close' in df.columns:
            st.warning("A coluna 'Adj Close' não foi encontrada. Utilizando 'Close'.")
            return df['Close']
        else:
            # Caso não haja nenhuma coluna válida, tenta usar a primeira coluna numérica
            for coluna in df.columns:
                if pd.api.types.is_numeric_dtype(df[coluna]):
                    st.warning(f"A coluna '{coluna}' foi usada como fallback para o preço.")
                    return df[coluna]
        
        st.error("Nenhuma coluna válida de preço ajustado ou fechado foi encontrada.")
        return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Erro ao baixar os dados de {tickers}: {e}")
        return pd.DataFrame()

# Função para obter dados do Banco Central (BCB)
def get_bcb(code):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados/ultimos/1?formato=json"
    try:
        r = requests.get(url)
        return float(r.json()[0]['valor'].replace(",", ".")) if r.status_code == 200 else None
    except Exception as e:
        st.error(f"Erro ao obter dados do Banco Central: {e}")
        return None

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
        # Verificando se a estrutura é um DataFrame e se o formato está correto
        if isinstance(dados, pd.DataFrame):
            # Caso os dados sejam um MultiIndex, tenta desembrulhar
            if isinstance(dados.columns, pd.MultiIndex):
                dados = dados.xs('Close', axis=1, level=0)
                st.warning("Usando preços de fechamento após desempacotar o MultiIndex.")
                
            st.success("Dados carregados com sucesso.")
            
            # Calculando os retornos diários
            retornos = dados.pct_change().dropna()

            # Verificando se existem valores nulos ou infinitos nos retornos
            if retornos.isnull().any().any() or np.isinf(retornos.values).any():
                st.warning("Os dados de retornos contêm valores inválidos ou ausentes. Verifique a qualidade dos dados.")
            else:
                # Exemplo de exibição de retorno
                st.success("Dados de retorno calculados com sucesso.")
                st.dataframe(retornos.head())

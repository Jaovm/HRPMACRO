import streamlit as st
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.covariance import LedoitWolf
import yfinance as yf
import requests

# Função para obter dados financeiros
def obter_preco_diario_ajustado(tickers):
    df = yf.download(tickers, start="2018-01-01", end="2025-01-01")['Close']
    return df

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

# Função para calcular o HRP (Hierarchical Risk Parity)
def calcular_hrp(tickers, retornos):
    # Calculando a matriz de covariância
    cov = LedoitWolf().fit(retornos).covariance_

    # Verificando se a covariância é válida
    if cov.size == 0:
        raise ValueError("A matriz de covariância está vazia ou inválida.")
    
    # Aplicando cluster hierárquico para determinar a estrutura de risco
    dist_matrix = sch.distance.pdist(cov)
    if dist_matrix.size == 0:
        raise ValueError("A matriz de distâncias está vazia. Verifique os dados.")
        
    linkage = sch.linkage(dist_matrix, method='ward')
    
    # Obtendo a estrutura hierárquica
    idx = sch.dendrogram(linkage, no_plot=True)['leaves']
    
    # Reorganizando a covariância de acordo com a hierarquia
    cov_sorted = cov[idx, :][:, idx]
    
    # Calculando os pesos
    n = len(tickers)
    pesos = np.ones(n) / n

    # Algoritmo HRP para redistribuir os pesos com base no risco hierárquico
    for i in range(n):
        pesos[i] = 1 / np.sqrt(np.diagonal(cov_sorted)[i])

    pesos = pesos / pesos.sum()
    return pesos

# Função para otimizar a carteira com o HRP
def otimizar_carteira_hrp(tickers, min_pct=0.01, max_pct=0.30, pesos_setor=None):
    # Obtendo os dados de preço ajustado
    dados = obter_preco_diario_ajustado(tickers)
    retornos = dados.pct_change().dropna()

    if retornos.isnull().any().any() or np.isinf(retornos.values).any():
        st.warning("Os dados de retornos contêm valores inválidos ou ausentes. Verifique a qualidade dos dados.")
        return None

    # Verificando se os dados de retornos são suficientes
    if retornos.shape[0] < 2:
        st.warning("Não há dados suficientes para calcular os retornos.")
        return None

    # Aplicando o método HRP
    pesos = calcular_hrp(tickers, retornos)
    
    # Ajustando os pesos para respeitar as restrições de alocação mínima e máxima
    pesos_ajustados = np.clip(pesos, min_pct, max_pct)
    pesos_ajustados = pesos_ajustados / pesos_ajustados.sum()  # Normalizando a soma para 1
    
    return pesos_ajustados

# Função para filtrar ativos válidos com base no cenário macroeconômico
def filtrar_ativos_validos(carteira, cenario):
    ativos_validos = []
    for ativo in carteira:
        # Lógica fictícia de validação de ativos com base no cenário macroeconômico
        if cenario == "Restritivo" and ativo == 'BBSE3.SA':
            ativos_validos.append({'ticker': ativo, 'preco_atual': 30, 'preco_alvo': 40, 'setor': 'Seguradoras'})
        elif cenario == "Expansionista" and ativo == 'ITUB3.SA':
            ativos_validos.append({'ticker': ativo, 'preco_atual': 45, 'preco_alvo': 50, 'setor': 'Bancos'})
    return ativos_validos

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
    ativos_validos = filtrar_ativos_validos(carteira, cenario)

    if not ativos_validos:
        st.warning("Nenhum ativo com preço atual abaixo do preço-alvo dos analistas.")
    else:
        tickers_validos = [a['ticker'] for a in ativos_validos]

        # Peso de cada setor baseado no cenário macroeconômico
        pesos_setor = {setor: 1 for setor in ['Seguradoras', 'Bancos']}  # Exemplo de setores

        try:
            # Otimizando a carteira com HRP
            pesos = otimizar_carteira_hrp(tickers_validos, pesos_setor=pesos_setor)
            if pesos is not None:
                # Calcula a nova alocação considerando o aporte
                aporte_total = aporte_mensal
                aporte_distribuido = pesos * aporte_total
                
                # Atualiza a tabela com os pesos atuais, novos e o aporte
                df_resultado = pd.DataFrame(ativos_validos)
                df_resultado["Alocação Atual (%)"] = 10  # Exemplo de alocação atual
                df_resultado["Alocação Nova (%)"] = (pesos * 100).round(2)
                df_resultado["Aporte (R$)"] = (aporte_distribuido).round(2)
                df_resultado = df_resultado.sort_values("Alocação Nova (%)", ascending=False)
                
                st.success("✅ Carteira otimizada com o método HRP.")
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

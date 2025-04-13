import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf


# Função para obter dados do Banco Central

# Configuração da página
st.set_page_config(page_title="Otimização de Carteira Inteligente", layout="wide", initial_sidebar_state="expanded")

# ====== CABEÇALHO ======
st.title("\U0001F4C8 Otimização e Sugestão de Carteira")
st.markdown("""
Este painel permite:
- **Análise macroeconômica automática** com dados do BCB e do mercado.
- **Filtragem de ações** com base em cenário, preço-alvo e exportação.
- **Otimização de carteira** via Sharpe e HRP.
---
""")

# ====== SIDEBAR ======
with st.sidebar:
    st.header("\U0001F9ED Navegação")
    pagina = st.radio("Selecione a etapa:", [
        "\U0001F4CC Introdução",
        "\U0001F310 Análise Macroeconômica",
        "\U0001F4C9 Otimização da Carteira",
        "\U0001F4B5 Sugestão de Aporte",
        "\u2705 Ranking de Ações"
    ])
    st.markdown("---")
    st.caption("Desenvolvido por [Seu Nome] 💼")

# ====== FUNÇÕES AUXILIARES (Defina estas funções em outro arquivo e importe) ======
# obter_macro()
def get_bcb(code):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados/ultimos/1?formato=json"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return float(r.json()[0]['valor'].replace(",", "."))
        else:
            st.warning(f"Não foi possível obter o dado com código {code} do BCB.")
            return None
    except Exception as e:
        st.error(f"Erro ao acessar dados do BCB (código {code}): {e}")
        return None

# Função para obter o preço atual do barril de petróleo (WTI)
def obter_preco_petroleo():
    try:
        dados = yf.Ticker("CL=F").history(period="5d")
        if not dados.empty and 'Close' in dados.columns:
            return float(dados['Close'].dropna().iloc[-1])
        else:
            return None
    except Exception as e:
        st.error(f"Erro ao obter preço do petróleo: {e}")
        return None

# Função principal que reúne os dados macroeconômicos
def obter_macro():
    return {
        "selic": get_bcb(432),       # Taxa Selic
        "ipca": get_bcb(433),        # IPCA
        "dolar": get_bcb(1),         # Dólar comercial
        "petroleo": obter_preco_petroleo()  # Preço do petróleo WTI
    }

# Função para classificar o cenário macroeconômico
def classificar_cenario_macro(m):
    if m['ipca'] > 5 or m['selic'] > 12:
        return "Restritivo"
    elif m['ipca'] < 4 and m['selic'] < 10:
        return "Expansionista"
    else:
        return "Neutro"

# filtrar_ativos_validos(tickers, cenario, macro)
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
                "score": score
            })

    ativos_validos.sort(key=lambda x: x['score'], reverse=True)
    return ativos_validos
    
# otimizar_carteira_sharpe(tickers)

def otimizar_carteira_sharpe(tickers):
    dados = obter_preco_diario_ajustado(tickers)
    retornos = dados.pct_change().dropna()
    media_retorno = retornos.mean()
    cov = LedoitWolf().fit(retornos).covariance_

    def sharpe_neg(pesos):
        retorno_esperado = np.dot(pesos, media_retorno)
        volatilidade = np.sqrt(np.dot(pesos.T, np.dot(cov, pesos)))
        return -retorno_esperado / volatilidade if volatilidade != 0 else 0

    n = len(tickers)
    pesos_iniciais = np.array([1/n] * n)
    limites = [(0, 1) for _ in range(n)]
    restricoes = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    resultado = minimize(sharpe_neg, pesos_iniciais, method='SLSQP', bounds=limites, constraints=restricoes)

    if resultado.success:
        return resultado.x
    else:
        return None

# otimizar_carteira_hrp(tickers)
def otimizar_carteira_hrp(tickers):
    dados = obter_preco_diario_ajustado(tickers)
    retornos = dados.pct_change().dropna()
    dist = np.sqrt(((1 - retornos.corr()) / 2).fillna(0))
    linkage_matrix = linkage(squareform(dist), method='single')

    def get_quasi_diag(link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0]*2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df1 = pd.Series(link[j, 1], index=i+1)
            sort_ix = pd.concat([sort_ix, df1])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    sort_ix = get_quasi_diag(linkage_matrix)
    sorted_tickers = [retornos.columns[i] for i in sort_ix]
    cov = LedoitWolf().fit(retornos).covariance_
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()

# ====== FUNÇÃO PRINCIPAL ======
def painel_inteligente():
    if pagina == "\U0001F4CC Introdução":
        st.subheader("Bem-vindo(a) à Otimização Inteligente de Carteira")
        st.markdown("""
        Este painel utiliza **dados macroeconômicos atualizados**, **preço-alvo dos analistas**, e técnicas modernas como **Hierarchical Risk Parity (HRP)** e **Otimização por Sharpe** para te ajudar a:

        - **Identificar oportunidades de compra**
        - **Sugerir alocações para novos aportes**
        - **Otimizar sua carteira com base no cenário econômico atual**
        ---
        """)

    elif pagina == "\U0001F310 Análise Macroeconômica":
        st.subheader("\U0001F30E Cenário Macroeconômico Atual")
        with st.spinner("Carregando dados macroeconômicos..."):
            macro = obter_macro()
            cenario = classificar_cenario_macro(macro)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Selic", f"{macro['selic']:.2f}%")
        col2.metric("IPCA", f"{macro['ipca']:.2f}%")
        col3.metric("Dólar", f"R$ {macro['dolar']:.2f}")
        col4.metric("Petróleo", f"US$ {macro['petroleo']:.2f}")

        st.success(f"Cenário Macroeconômico Classificado como: `{cenario}`")

    elif pagina == "\U0001F4C9 Otimização da Carteira":
        st.subheader("\u2696\ufe0f Otimização com HRP")
        carteira_usuario = st.text_input("Tickers da sua carteira (separados por vírgula):")
        if carteira_usuario:
            tickers = [t.strip().upper() for t in carteira_usuario.split(',')]
            with st.spinner("Calculando alocação ideal com HRP..."):
                pesos = otimizar_carteira_hrp(tickers)
            if pesos is not None:
                df = pd.DataFrame({"Ticker": pesos.index, "Peso (%)": np.round(pesos.values * 100, 2)})
                st.dataframe(df.reset_index(drop=True))
            else:
                st.error("Erro na otimização da carteira.")

    elif pagina == "\U0001F4B5 Sugestão de Aporte":
        st.subheader("Sugestão de Aporte com Base no Cenário Atual")
        carteira_input = st.text_input("Digite os tickers da sua carteira:")
        if carteira_input:
            tickers = [t.strip().upper() for t in carteira_input.split(',')]
            macro = obter_macro()
            cenario = classificar_cenario_macro(macro)
            recomendadas = filtrar_ativos_validos(tickers, cenario, macro)
            if recomendadas:
                df = pd.DataFrame(recomendadas)
                st.dataframe(df[['ticker', 'setor', 'preco_atual', 'preco_alvo', 'favorecido', 'score']])
            else:
                st.warning("Nenhuma ação recomendada com base no cenário atual.")

    elif pagina == "✅ Ranking de Ações":
        st.subheader("\U0001F3C6 Ranking de Ações da sua Carteira")
        carteira_input = st.text_input("Digite os tickers da sua carteira:")
        if carteira_input:
            tickers = [t.strip().upper() for t in carteira_input.split(',')]
            macro = obter_macro()
            cenario = classificar_cenario_macro(macro)
            ranking = filtrar_ativos_validos(tickers, cenario, macro)
            if ranking:
                df_ranking = pd.DataFrame(ranking)
                st.dataframe(df_ranking[['ticker', 'score', 'preco_atual', 'preco_alvo', 'setor', 'favorecido']])
            else:
                st.warning("Nenhum ativo ranqueado com os dados atuais.")

# ====== EXECUÇÃO ======
painel_inteligente()

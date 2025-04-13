import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.optimize import minimize

# ========= DICIONÁRIOS ==========

setores_por_ticker = {
    # Bancos
    'ITUB4.SA': 'Bancos',
    'BBDC4.SA': 'Bancos',
    'SANB11.SA': 'Bancos',
    'BBAS3.SA': 'Bancos',
    'ABCB4.SA': 'Bancos',
    'BRSR6.SA': 'Bancos',
    'BMGB4.SA': 'Bancos',
    'BPAC11.SA': 'Bancos',

    # Seguradoras
    'BBSE3.SA': 'Seguradoras',
    'PSSA3.SA': 'Seguradoras',
    'SULA11.SA': 'Seguradoras',
    'CXSE3.SA': 'Seguradoras',

    # Bolsas e Serviços Financeiros
    'B3SA3.SA': 'Bolsas e Serviços Financeiros',
    'XPBR31.SA': 'Bolsas e Serviços Financeiros',

    # Energia Elétrica
    'EGIE3.SA': 'Energia Elétrica',
    'CPLE6.SA': 'Energia Elétrica',
    'TAEE11.SA': 'Energia Elétrica',
    'CMIG4.SA': 'Energia Elétrica',
    'AURE3.SA': 'Energia Elétrica',
    'CPFE3.SA': 'Energia Elétrica',
    'AESB3.SA': 'Energia Elétrica',

    # Petróleo, Gás e Biocombustíveis
    'PETR4.SA': 'Petróleo, Gás e Biocombustíveis',
    'PRIO3.SA': 'Petróleo, Gás e Biocombustíveis',
    'RECV3.SA': 'Petróleo, Gás e Biocombustíveis',
    'RRRP3.SA': 'Petróleo, Gás e Biocombustíveis',
    'UGPA3.SA': 'Petróleo, Gás e Biocombustíveis',
    'VBBR3.SA': 'Petróleo, Gás e Biocombustíveis',

    # Mineração e Siderurgia
    'VALE3.SA': 'Mineração e Siderurgia',
    'CSNA3.SA': 'Mineração e Siderurgia',
    'GGBR4.SA': 'Mineração e Siderurgia',
    'CMIN3.SA': 'Mineração e Siderurgia',
    'GOAU4.SA': 'Mineração e Siderurgia',
    'BRAP4.SA': 'Mineração e Siderurgia',

    # Indústria e Bens de Capital
    'WEGE3.SA': 'Indústria e Bens de Capital',
    'RANI3.SA': 'Indústria e Bens de Capital',
    'KLBN11.SA': 'Indústria e Bens de Capital',
    'SUZB3.SA': 'Indústria e Bens de Capital',
    'UNIP6.SA': 'Indústria e Bens de Capital',
    'KEPL3.SA': 'Indústria e Bens de Capital',

    # Agronegócio
    'AGRO3.SA': 'Agronegócio',
    'SLCE3.SA': 'Agronegócio',
    'SMTO3.SA': 'Agronegócio',
    'CAML3.SA': 'Agronegócio',

    # Saúde
    'HAPV3.SA': 'Saúde',
    'FLRY3.SA': 'Saúde',
    'RDOR3.SA': 'Saúde',
    'QUAL3.SA': 'Saúde',
    'RADL3.SA': 'Saúde',

    # Tecnologia
    'TOTS3.SA': 'Tecnologia',
    'POSI3.SA': 'Tecnologia',
    'LINX3.SA': 'Tecnologia',
    'LWSA3.SA': 'Tecnologia',

    # Consumo Discricionário
    'MGLU3.SA': 'Consumo Discricionário',
    'LREN3.SA': 'Consumo Discricionário',
    'RENT3.SA': 'Consumo Discricionário',
    'ARZZ3.SA': 'Consumo Discricionário',
    'ALPA4.SA': 'Consumo Discricionário',

    # Consumo Básico
    'ABEV3.SA': 'Consumo Básico',
    'NTCO3.SA': 'Consumo Básico',
    'PCAR3.SA': 'Consumo Básico',
    'MDIA3.SA': 'Consumo Básico',

    # Comunicação
    'VIVT3.SA': 'Comunicação',
    'TIMS3.SA': 'Comunicação',
    'OIBR3.SA': 'Comunicação',

    # Utilidades Públicas
    'SBSP3.SA': 'Utilidades Públicas',
    'SAPR11.SA': 'Utilidades Públicas',
    'CSMG3.SA': 'Utilidades Públicas',
    'ALUP11.SA': 'Utilidades Públicas',
    'CPLE6.SA': 'Utilidades Públicas',
}


setores_por_cenario = {
    "Expansionista": [
        'Consumo Discricionário',
        'Tecnologia',
        'Indústria e Bens de Capital',
        'Agronegócio'
    ],
    "Neutro": [
        'Saúde',
        'Bancos',
        'Seguradoras',
        'Bolsas e Serviços Financeiros',
        'Utilidades Públicas'
    ],
    "Restritivo": [
        'Energia Elétrica',
        'Petróleo, Gás e Biocombustíveis',
        'Mineração e Siderurgia',
        'Consumo Básico',
        'Comunicação'
    ]
}

empresas_exportadoras = [
    'VALE3.SA',  # Mineração
    'SUZB3.SA',  # Celulose
    'KLBN11.SA', # Papel e Celulose
    'AGRO3.SA',  # Agronegócio
    'PRIO3.SA',  # Petróleo
    'SLCE3.SA',  # Agronegócio
    'SMTO3.SA',  # Açúcar e Etanol
    'CSNA3.SA',  # Siderurgia
    'GGBR4.SA',  # Siderurgia
    'CMIN3.SA',  # Mineração
]


# ========= MACRO ==========
def get_bcb(code):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados/ultimos/1?formato=json"
    r = requests.get(url)
    return float(r.json()[0]['valor'].replace(",", ".")) if r.status_code == 200 else None

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

def obter_preco_diario_ajustado(tickers, periodo='7y'):
    dados = yf.download(tickers, period=periodo, group_by='ticker', auto_adjust=True)
    if len(tickers) == 1:
        return dados['Adj Close'].to_frame()
    else:
        return dados['Adj Close']


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
                "score": score
            })

    ativos_validos.sort(key=lambda x: x['score'], reverse=True)
    return ativos_validos

# ========= OTIMIZAÇÃO ==========
def obter_preco_diario_ajustado(tickers):
    dados_brutos = yf.download(tickers, period="3y", auto_adjust=False)

    if isinstance(dados_brutos.columns, pd.MultiIndex):
        if 'Adj Close' in dados_brutos.columns.get_level_values(0):
            return dados_brutos['Adj Close']
        elif 'Close' in dados_brutos.columns.get_level_values(0):
            return dados_brutos['Close']
        else:
            raise ValueError("Colunas 'Adj Close' ou 'Close' não encontradas nos dados.")
    else:
        if 'Adj Close' in dados_brutos.columns:
            return dados_brutos[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
        elif 'Close' in dados_brutos.columns:
            return dados_brutos[['Close']].rename(columns={'Close': tickers[0]})
        else:
            raise ValueError("Coluna 'Adj Close' ou 'Close' não encontrada nos dados.")

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

    def get_cluster_var(cov, cluster_items):
        cov_slice = cov[np.ix_(cluster_items, cluster_items)]
        w_ = 1. / np.diag(cov_slice)
        w_ /= w_.sum()
        return np.dot(w_, np.dot(cov_slice, w_))

    def recursive_bisection(cov, sort_ix):
        w = pd.Series(1, index=sort_ix)
        cluster_items = [sort_ix]
        while len(cluster_items) > 0:
            cluster_items = [i[j:k] for i in cluster_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(cluster_items), 2):
                c0 = cluster_items[i]
                c1 = cluster_items[i + 1]
                var0 = get_cluster_var(cov, c0)
                var1 = get_cluster_var(cov, c1)
                alpha = 1 - var0 / (var0 + var1)
                w[c0] *= alpha
                w[c1] *= 1 - alpha
        return w

    hrp_weights = recursive_bisection(cov, list(range(len(tickers))))
    return hrp_weights.values
# Painel Principal
st.set_page_config(page_title="Otimização de Carteira", layout="wide")
st.title("\U0001F4C8 Otimização e Sugestão de Carteira")

st.markdown("""
Este painel permite:
- **Análise macroeconômica automática** com dados do BCB e do mercado.
- **Filtragem de ações** com base em cenário, preço-alvo e exportação.
- **Otimização de carteira** via Sharpe e HRP.

---
""")

# Abas do Painel
with st.sidebar:
    st.header("🧭 Navegação")
    pagina = st.radio("Selecione a etapa:", [
        "📌 Introdução",
        "🌐 Análise Macroeconômica",
        "📈 Sugestão de Aporte",
        "⚙️ Otimização da Carteira",
        "✅ Ranking de Ações"
    ])
    st.markdown("---")
    st.caption("Desenvolvido por [Seu Nome] 💼")

# ====== FUNÇÃO PRINCIPAL ======
def painel_inteligente():
    if pagina == "📌 Introdução":
        st.subheader("Bem-vindo(a) à Otimização Inteligente de Carteira")
        st.markdown("""
        Este painel utiliza **dados macroeconômicos atualizados**, **preço-alvo dos analistas**, e técnicas modernas como **Hierarchical Risk Parity (HRP)** e **Otimização por Sharpe** para te ajudar a:

        - **Identificar oportunidades de compra**
        - **Sugerir alocações para novos aportes**
        - **Otimizar sua carteira com base no cenário econômico atual**

        ---
        """)

    elif pagina == "🌐 Análise Macroeconômica":
        st.subheader("🌎 Cenário Macroeconômico Atual")

        with st.spinner("🔄 Carregando dados macroeconômicos..."):
            macro = obter_macro()
            cenario = classificar_cenario_macro(macro)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📉 Selic", f"{macro['selic']:.2f}%")
        col2.metric("📈 IPCA", f"{macro['ipca']:.2f}%")
        col3.metric("💵 Dólar", f"R$ {macro['dolar']:.2f}")
        col4.metric("🛢️ Petróleo (Brent)", f"US$ {macro['petroleo']:.2f}")

        st.success(f"**Cenário Macroeconômico Classificado como: `{cenario}`**")

        st.markdown("Com base nesse cenário, alguns setores tendem a se destacar mais que outros. Utilize essa informação para orientar seus investimentos.")

    elif pagina == "📈 Sugestão de Aporte":
        st.subheader("💡 Sugestão de Aporte com Base no Cenário Atual")

        carteira_usuario = st.text_input("Digite os tickers da sua carteira separados por vírgula (ex: PETR4.SA,VALE3.SA,...):")
        if carteira_usuario:
            tickers = [t.strip().upper() for t in carteira_usuario.split(',')]

            with st.spinner("🔍 Analisando ações..."):
                macro = obter_macro()
                cenario = classificar_cenario_macro(macro)
                recomendadas = filtrar_ativos_validos(tickers, cenario, macro)

            if recomendadas:
                st.success(f"🎯 {len(recomendadas)} ações recomendadas para aporte:")
                df_rec = pd.DataFrame(recomendadas)
                st.dataframe(df_rec[['ticker', 'setor', 'preco_atual', 'preco_alvo', 'favorecido', 'score']])
            else:
                st.warning("Nenhuma ação da sua carteira apresentou potencial interessante com base nos critérios definidos.")

    elif pagina == "⚙️ Otimização da Carteira":
        st.subheader("⚖️ Otimização com Hierarchical Risk Parity (HRP)")

        carteira_usuario = st.text_input("Digite os tickers da sua carteira para otimização:", key="otimizacao")
        if carteira_usuario:
            tickers = [t.strip().upper() for t in carteira_usuario.split(',')]
            try:
                with st.spinner("📈 Calculando alocação ótima com HRP..."):
                    pesos_otimizados = otimizar_carteira_hrp(tickers)

                if pesos_otimizados is not None:
                    df_pesos = pd.DataFrame({
                        'Ticker': tickers,
                        'Peso Otimizado (%)': np.round(pesos_otimizados * 100, 2)
                    }).sort_values(by='Peso Otimizado (%)', ascending=False)
                    st.success("✅ Otimização concluída com sucesso!")
                    st.dataframe(df_pesos.reset_index(drop=True))
                else:
                    st.error("❌ Não foi possível otimizar a carteira.")
            except Exception as e:
                st.error(f"Erro durante a otimização: {e}")

    elif pagina == "✅ Ranking de Ações":
        st.subheader("🏆 Ranking de Ações da sua Carteira")
        carteira_usuario = st.text_input("Digite seus tickers:", key="ranking")
        if carteira_usuario:
            tickers = [t.strip().upper() for t in carteira_usuario.split(',')]

            with st.spinner("📊 Calculando score de cada ativo..."):
                macro = obter_macro()
                cenario = classificar_cenario_macro(macro)
                ranking = filtrar_ativos_validos(tickers, cenario, macro)

            if ranking:
                df_ranking = pd.DataFrame(ranking)
                st.dataframe(df_ranking[['ticker', 'score', 'preco_atual', 'preco_alvo', 'setor', 'favorecido']])
            else:
                st.warning("Nenhum ativo pôde ser ranqueado com os dados disponíveis.")

# ====== EXECUÇÃO ======
painel_inteligente()

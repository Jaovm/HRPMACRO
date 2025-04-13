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
aba = st.sidebar.radio("Selecione uma seção:", [
    "1. Análise Macroeconômica",
    "2. Filtrar Ações por Cenário",
    "3. Otimizar Carteira (Sharpe)",
    "4. Otimizar Carteira (HRP)",
    "5. Sugestão de Aporte"
])

if aba == "1. Análise Macroeconômica":
    st.header("\U0001F4CA Indicadores Macroeconômicos")
    macro = obter_macro()
    cenario = classificar_cenario_macro(macro)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SELIC (%)", f"{macro['selic']:.2f}")
    col2.metric("IPCA (%)", f"{macro['ipca']:.2f}")
    col3.metric("Dólar (R$)", f"{macro['dolar']:.2f}")
    col4.metric("Petróleo (US$)", f"{macro['petroleo']:.2f}")

    st.success(f"**Cenário Macroeconômico Classificado:** {cenario}")

elif aba == "2. Filtrar Ações por Cenário":
    st.header("\U0001F50D Ações Interessantes para Compra")
    tickers_input = st.text_area("Digite os tickers da sua carteira, separados por vírgula:",
                                 value="AGRO3.SA,BBAS3.SA,BBSE3.SA,BPAC11.SA,EGIE3.SA,ITUB4.SA,PRIO3.SA,PSSA3.SA,SAPR11.SA,SBSP3.SA,VIVT3.SA,WEGE3.SA,TOTS3.SA,B3SA3.SA,TAEE11.SA")
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip() != ""]

    macro = obter_macro()
    cenario = classificar_cenario_macro(macro)
    ativos_validos = filtrar_ativos_validos(tickers, cenario, macro)

    if ativos_validos:
        df_resultado = pd.DataFrame(ativos_validos)
        st.dataframe(df_resultado.style.format({"preco_atual": "R$ {:.2f}", "preco_alvo": "R$ {:.2f}", "score": "{:.2%}"}))
    else:
        st.warning("Nenhuma ação elegível encontrada com base no cenário atual.")

elif aba == "3. Otimizar Carteira (Sharpe)":
    st.header("\U0001F4C9 Otimização via Índice de Sharpe")
    tickers = st.text_input("Tickers da carteira (separados por vírgula):", value="AGRO3.SA,BBAS3.SA,PRIO3.SA,PSSA3.SA,WEGE3.SA")
    tickers = [t.strip() for t in tickers.split(",") if t.strip() != ""]

    if st.button("Otimizar"):
        pesos = otimizar_carteira_sharpe(tickers)
        if pesos is not None:
            df = pd.DataFrame({"Ticker": tickers, "Peso (%)": np.round(pesos * 100, 2)})
            st.dataframe(df)
        else:
            st.error("Erro na otimização da carteira.")

elif aba == "4. Otimizar Carteira (HRP)":
    st.header("\U0001F4C9 Otimização via HRP (Hierarchical Risk Parity)")
    tickers = st.text_input("Tickers da carteira (separados por vírgula):", value="AGRO3.SA,BBAS3.SA,PRIO3.SA,PSSA3.SA,WEGE3.SA")
    tickers = [t.strip() for t in tickers.split(",") if t.strip() != ""]

    if st.button("Rodar HRP"):
        pesos = otimizar_carteira_hrp(tickers)
        if pesos is not None:
            df = pd.DataFrame({"Ticker": pesos.index, "Peso (%)": np.round(pesos.values * 100, 2)})
            st.dataframe(df)
        else:
            st.error("Erro na otimização com HRP.")

elif aba == "5. Sugestão de Aporte":
    st.header("\U0001F4B5 Sugestão de Aporte com Base no Cenário")
    carteira_input = st.text_area("Tickers e alocações atuais (ex: AGRO3.SA:10,PRIO3.SA:15,...):",
                                  value="AGRO3.SA:10,BBAS3.SA:1.2,BBSE3.SA:6.5,BPAC11.SA:10.6,EGIE3.SA:5,ITUB4.SA:0.5,PRIO3.SA:15,PSSA3.SA:15,SAPR11.SA:6.7,SBSP3.SA:4,VIVT3.SA:6.4,WEGE3.SA:15,TOTS3.SA:1,B3SA3.SA:0.1,TAEE11.SA:3")

    aporte = st.number_input("Valor do novo aporte (R$):", min_value=100.0, step=100.0)

    if st.button("Sugerir alocação do aporte"):
        st.info("Funcionalidade a ser integrada com algoritmo HRP, filtros de cenário e restrições.")

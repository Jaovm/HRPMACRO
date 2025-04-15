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
    'ITSA4.SA': 'Bancos',

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
    'SAPR11.SA': 'Utilidades Públicas'
    'SAPR3.SA': 'Utilidades Públicas'
    'SAPR4.SA': 'Utilidades Públicas',
    'CSMG3.SA': 'Utilidades Públicas',
    'ALUP11.SA': 'Utilidades Públicas',
    'CPLE6.SA': 'Utilidades Públicas',

    # Adicionando ativos novos conforme solicitado
    'CRFB3.SA': 'Consumo Discricionário',
    'COGN3.SA': 'Tecnologia',
    'OIBR3.SA': 'Comunicação',
    'CCRO3.SA': 'Utilidades Públicas',
    'BEEF3.SA': 'Consumo Discricionário',
    'AZUL4.SA': 'Consumo Discricionário',
    'POMO4.SA': 'Indústria e Bens de Capital',
    'RAIL3.SA': 'Indústria e Bens de Capital',
    'CVCB3.SA': 'Consumo Discricionário',
    'BRAV3.SA': 'Bancos',
    'PETR3.SA': 'Petróleo, Gás e Biocombustíveis',
    'VAMO3.SA': 'Consumo Discricionário',
    'CSAN3.SA': 'Energia Elétrica',
    'USIM5.SA': 'Mineração e Siderurgia',
    'RAIZ4.SA': 'Agronegócio',
    'ELET3.SA': 'Energia Elétrica',
    'CMIG4.SA': 'Energia Elétrica',
    'EQTL3.SA': 'Energia Elétrica',
    'ANIM3.SA': 'Saúde',
    'MRVE3.SA': 'Consumo Discricionário',
    'AMOB3.SA': 'Tecnologia',
    'RAPT4.SA': 'Consumo Discricionário',
    'CSNA3.SA': 'Mineração e Siderurgia',
    'RENT3.SA': 'Consumo Discricionário',
    'MRFG3.SA': 'Consumo Básico',
    'JBSS3.SA': 'Consumo Básico',
    'VBBR3.SA': 'Petróleo, Gás e Biocombustíveis',
    'BBDC3.SA': 'Bancos',
    'IFCM3.SA': 'Tecnologia',
    'BHIA3.SA': 'Bancos',
    'LWSA3.SA': 'Tecnologia',
    'SIMH3.SA': 'Saúde',
    'CMIN3.SA': 'Mineração e Siderurgia',
    'UGPA3.SA': 'Petróleo, Gás e Biocombustíveis',
    'MOVI3.SA': 'Consumo Discricionário',
    'GFSA3.SA': 'Consumo Discricionário',
    'AZEV4.SA': 'Saúde',
    'RADL3.SA': 'Saúde',
    'BPAC11.SA': 'Bancos',
    'PETZ3.SA': 'Saúde',
    'AURE3.SA': 'Energia Elétrica',
    'ENEV3.SA': 'Energia Elétrica',
    'WEGE3.SA': 'Indústria e Bens de Capital',
    'CPLE3.SA': 'Energia Elétrica',
    'SRNA3.SA': 'Indústria e Bens de Capital',
    'BRFS3.SA': 'Consumo Básico',
    'SLCE3.SA': 'Agronegócio',
    'CBAV3.SA': 'Consumo Básico',
    'ECOR3.SA': 'Tecnologia',
    'KLBN11.SA': 'Indústria e Bens de Capital',
    'EMBR3.SA': 'Indústria e Bens de Capital',
    'MULT3.SA': 'Bancos',
    'CYRE3.SA': 'Indústria e Bens de Capital',
    'RDOR3.SA': 'Saúde',
    'TIMS3.SA': 'Comunicação',
    'SUZB3.SA': 'Indústria e Bens de Capital',
    'ALOS3.SA': 'Saúde',
    'SMFT3.SA': 'Tecnologia',
    'FLRY3.SA': 'Saúde',
    'IGTI11.SA': 'Tecnologia',
    'AMER3.SA': 'Consumo Discricionário',
    'YDUQ3.SA': 'Tecnologia',
    'STBP3.SA': 'Bancos',
    'GMAT3.SA': 'Indústria e Bens de Capital',
    'TOTS3.SA': 'Tecnologia',
    'CEAB3.SA': 'Indústria e Bens de Capital',
    'EZTC3.SA': 'Consumo Discricionário',
    'BRAP4.SA': 'Mineração e Siderurgia',
    'RECV3.SA': 'Petróleo, Gás e Biocombustíveis',
    'VIVA3.SA': 'Saúde',
    'DXCO3.SA': 'Tecnologia',
    'SANB11.SA': 'Bancos',
    'BBSE3.SA': 'Seguradoras',
    'LJQQ3.SA': 'Tecnologia',
    'PMAM3.SA': 'Saúde',
    'SBSP3.SA': 'Utilidades Públicas',
    'ENGI11.SA': 'Energia Elétrica',
    'JHSF3.SA': 'Indústria e Bens de Capital',
    'INTB3.SA': 'Indústria e Bens de Capital',
    'RCSL4.SA': 'Tecnologia',
    'GOLL4.SA': 'Consumo Discricionário',
    'CXSE3.SA': 'Seguradoras',
    'QUAL3.SA': 'Saúde',
    'BRKM5.SA': 'Indústria e Bens de Capital',
    'HYPE3.SA': 'Saúde',
    'IRBR3.SA': 'Tecnologia',
    'MDIA3.SA': 'Consumo Básico',
    'BEEF3.SA': 'Consumo Discricionário',
    'MMXM3.SA': 'Indústria e Bens de Capital',
    'USIM5.SA': 'Mineração e Siderurgia',
}


setores_por_cenario = {
    "Expansão Forte": [
        'Consumo Discricionário', 'Tecnologia',
        'Indústria e Bens de Capital', 'Agronegócio'
    ],
    "Expansão Moderada": [
        'Consumo Discricionário', 'Tecnologia',
        'Indústria e Bens de Capital', 'Agronegócio', 'Saúde'
    ],
    "Estável": [
        'Saúde', 'Bancos', 'Seguradoras',
        'Bolsas e Serviços Financeiros', 'Utilidades Públicas'
    ],
    "Contração Moderada": [
        'Energia Elétrica', 'Petróleo, Gás e Biocombustíveis',
        'Mineração e Siderurgia', 'Consumo Básico', 'Comunicação'
    ],
    "Contração Forte": [
        'Energia Elétrica', 'Petróleo, Gás e Biocombustíveis',
        'Mineração e Siderurgia', 'Consumo Básico'
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

def get_ipca_anualizado():
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados/ultimos/12?formato=json"
    r = requests.get(url)
    if r.status_code == 200:
        dados = r.json()
        valores = [float(d['valor'].replace(",", ".")) for d in dados]
        return sum(valores)
    return None


def obter_preco_commodity(ticker, nome="Commodity"):
    try:
        dados = yf.Ticker(ticker).history(period="5d")
        if not dados.empty and 'Close' in dados.columns:
            preco = dados['Close'].dropna().iloc[-1]
            return float(preco)
        else:
            st.warning(f"Preço indisponível para {nome}.")
            return None
    except Exception as e:
        st.error(f"Erro ao obter preço para {nome} ({ticker}): {e}")
        return None


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
        "ipca": get_ipca_anualizado(),
        "dolar": get_bcb(1),
        "pib": get_bcb(4380),  # PIB anual
        "petroleo": obter_preco_petroleo(),
        "minerio": obter_preco_commodity("TIO=F", "Minério de Ferro (proxy)"),
        "soja": obter_preco_commodity("ZS=F", "Soja"),
        "milho": obter_preco_commodity("ZC=F", "Milho")
    }

def pontuar_macro(m):
    print(f"Pontuando macro: {m}")
    score = 0

    # Selic
    if m.get('selic') is not None:
        if m['selic'] < 9:
            score += 2  # cenário expansionista
        elif m['selic'] <= 11:
            score += 1
        elif m['selic'] <= 13:
            score += 0
        else:
            score -= 1  # cenário restritivo

    # IPCA (assumindo anualizado)
    if m.get('ipca') is not None:
        if m['ipca'] < 3:
            score += 1
        elif m['ipca'] <= 5:
            score += 0
        else:
            score -= 1

    # Dólar
    if m.get('dolar') is not None:
        if m['dolar'] < 4.8:
            score += 1
        elif m['dolar'] <= 5.3:
            score += 0
        else:
            score -= 1

    # PIB
    if m.get('pib') is not None:
        if m['pib'] > 2:
            score += 2
        elif m['pib'] > 0:
            score += 1
        else:
            score -= 1

    # Soja e Milho (valores em R$/saca 60kg, convertido se necessário)
    if m.get('soja') and m.get('milho'):
        media_agro = (m['soja'] / 1000 + m['milho'] / 1000) / 2
        if media_agro > 1.3:
            score += 1
        elif media_agro > 1.0:
            score += 0
        else:
            score -= 1

    # Minério de ferro (TIO=F, referência em USD/ton)
    if m.get('minerio') is not None:
        if m['minerio'] > 120:
            score += 1
        elif m['minerio'] >= 90:
            score += 0
        else:
            score -= 1

    print(f"Score macro calculado: {score}")
    return score



def classificar_cenario_macro(m):
    score = pontuar_macro(m)
    if score >= 5:
        return "Expansão Forte"
    elif score >= 3:
        return "Expansão Moderada"
    elif score >= 0:
        return "Estável"
    elif score >= -2:
        return "Contração Moderada"
    else:
        return "Contração Forte"


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
# Novo modelo com commodities separadas
sensibilidade_setorial = {
    'Bancos':                          {'juros': 1,  'inflação': 0,  'cambio': 0,  'pib': 1,  'commodities_agro': 1, 'commodities_minerio': 1},
    'Seguradoras':                     {'juros': 2,  'inflação': 0,  'cambio': 0,  'pib': 1,  'commodities_agro': 0, 'commodities_minerio': 0},
    'Bolsas e Serviços Financeiros':  {'juros': 1,  'inflação': 0,  'cambio': 0,  'pib': 2,  'commodities_agro': 0, 'commodities_minerio': 0},
    'Energia Elétrica':               {'juros': 2,  'inflação': 1,  'cambio': -1, 'pib': -1, 'commodities_agro': -1, 'commodities_minerio': -1},
    'Petróleo, Gás e Biocombustíveis':{'juros': 0,  'inflação': 0,  'cambio': 2,  'pib': 1,  'commodities_agro': 0,  'commodities_minerio': 0},
    'Mineração e Siderurgia':         {'juros': 0,  'inflação': 0,  'cambio': 2,  'pib': 1,  'commodities_agro': 0,  'commodities_minerio': 2},
    'Indústria e Bens de Capital':    {'juros': -1, 'inflação': -1, 'cambio': -1, 'pib': 2,  'commodities_agro': 0,  'commodities_minerio': 0},
    'Agronegócio':                    {'juros': 0,  'inflação': -1, 'cambio': 2,  'pib': 1,  'commodities_agro': 2,  'commodities_minerio': 0},
    'Saúde':                          {'juros': 0,  'inflação': 0,  'cambio': 0,  'pib': 1,  'commodities_agro': 0,  'commodities_minerio': 0},
    'Tecnologia':                     {'juros': -2, 'inflação': 0,  'cambio': 0,  'pib': 2,  'commodities_agro': -1, 'commodities_minerio': -1},
    'Consumo Discricionário':         {'juros': -2, 'inflação': -1, 'cambio': -1, 'pib': 2,  'commodities_agro': -1, 'commodities_minerio': -1},
    'Consumo Básico':                 {'juros': 1,  'inflação': -2, 'cambio': -1, 'pib': 1,  'commodities_agro': -1, 'commodities_minerio': -1},
    'Comunicação':                    {'juros': 0,  'inflação': 0,  'cambio': -1, 'pib': 1,  'commodities_agro': 0,  'commodities_minerio': 0},
    'Utilidades Públicas':            {'juros': 2,  'inflação': 1,  'cambio': -1, 'pib': -1, 'commodities_agro': -1, 'commodities_minerio': -1}
}

def calcular_score(preco_atual, preco_alvo, favorecido, ticker, macro, usar_pesos_macroeconomicos=True):
    upside = (preco_alvo - preco_atual) / preco_atual
    bonus = 0.1 if favorecido else 0

    setor = setores_por_ticker.get(ticker)
    score_macro = 0

    if setor in sensibilidade_setorial and usar_pesos_macroeconomicos:  # Verifique se devemos usar os pesos macroeconômicos
        s = sensibilidade_setorial[setor]

        if macro['selic'] is not None:
            score_macro += s['juros'] * (1 if macro['selic'] > 10 else -1)
        if macro['ipca'] is not None:
            score_macro += s['inflação'] * (1 if macro['ipca'] > 5 else -1)
        if macro['dolar'] is not None:
            score_macro += s['cambio'] * (1 if macro['dolar'] > 5 else -1)
        if macro['pib'] is not None:
            score_macro += s['pib'] * (1 if macro['pib'] > 0 else -1)

        if macro['soja'] is not None and macro['milho'] is not None:
            media_agro = (macro['soja'] / 1000 + macro['milho'] / 1000) / 2
            score_macro += s.get('commodities_agro', 0) * (1 if media_agro > 1 else -1)

        if macro['minerio'] is not None:
            score_macro += s.get('commodities_minerio', 0) * (1 if macro['minerio'] > 100 else -1)

    if ticker in empresas_exportadoras:
        if macro['dolar'] and macro['dolar'] > 5:
            bonus += 0.05
        if macro['petroleo'] and macro['petroleo'] > 80:
            bonus += 0.05

    score_total = upside + bonus + 0.01 * score_macro
    return score_total


def filtrar_ativos_validos(carteira, cenario, macro, usar_pesos_macroeconomicos=True):
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
            score = calcular_score(preco_atual, preco_alvo, favorecido, ticker, macro, usar_pesos_macroeconomicos)
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



# ========= STREAMLIT ==========
st.set_page_config(page_title="Sugestão de Carteira", layout="wide")
st.title("📊 Sugestão e Otimização de Carteira com Base no Cenário Macroeconômico")

macro = obter_macro()
cenario = classificar_cenario_macro(macro)
score_macro = pontuar_macro(macro)
st.markdown(f"### 🧭 Cenário Macroeconômico Atual: **{cenario}** (Score: {score_macro})")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Selic (%)", f"{macro['selic']:.2f}")
col2.metric("Inflação IPCA (%)", f"{macro['ipca']:.2f}")
col3.metric("Dólar (R$)", f"{macro['dolar']:.2f}")
col4.metric("Petróleo (US$)", f"{macro['petroleo']:.2f}" if macro['petroleo'] else "N/A")
st.info(f"**Cenário Macroeconômico Atual:** {cenario}")

import streamlit as st
import numpy as np

# --- SIDEBAR ---
with st.sidebar:
    st.header("Parâmetros")
    st.markdown("### Dados dos Ativos")

    # Tickers e pesos default
    tickers_default = [
        "AGRO3.SA", "BBAS3.SA", "BBSE3.SA", "BPAC11.SA", "EGIE3.SA",
        "ITUB3.SA", "PRIO3.SA", "PSSA3.SA", "SAPR3.SA", "SBSP3.SA",
        "VIVT3.SA", "WEGE3.SA", "TOTS3.SA", "B3SA3.SA", "TAEE3.SA"
    ]
    pesos_default = [
        0.10, 0.012, 0.065, 0.106, 0.05,
        0.005, 0.15, 0.15, 0.067, 0.04,
        0.064, 0.15, 0.01, 0.001, 0.03
    ]

    # Número de ativos controlado por estado da sessão
    if "num_ativos" not in st.session_state:
        st.session_state.num_ativos = len(tickers_default)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("( + )", key="add_ativo"):
            st.session_state.num_ativos += 1
    with col2:
        if st.button("( - )", key="remove_ativo") and st.session_state.num_ativos > 1:
            st.session_state.num_ativos -= 1

    # Lista para armazenar inputs
    tickers = []
    pesos = []

    for i in range(st.session_state.num_ativos):
        col1, col2 = st.columns(2)
        with col1:
            ticker_default = tickers_default[i] if i < len(tickers_default) else ""
            ticker = st.text_input(f"Ticker do Ativo {i+1}", value=ticker_default, key=f"ticker_{i}").upper()
        with col2:
            peso_default = pesos_default[i] if i < len(pesos_default) else 1.0
            peso = st.number_input(f"Peso do Ativo {i+1}", min_value=0.0, step=0.01, value=peso_default, key=f"peso_{i}")
        if ticker:
            tickers.append(ticker)
            pesos.append(peso)

    # Normalização
    pesos_array = np.array(pesos)
    if pesos_array.sum() > 0:
        pesos_atuais = pesos_array / pesos_array.sum()
    else:
        st.error("A soma dos pesos deve ser maior que 0.")
        st.stop()

# Constrói a carteira com os tickers e pesos normalizados
carteira = dict(zip(tickers, pesos_atuais))



aporte = st.number_input("💰 Valor do aporte mensal (R$)", min_value=100.0, value=1000.0, step=100.0)
usar_hrp = st.checkbox("Utilizar HRP em vez de Sharpe máximo")
usar_pesos_macroeconomicos = st.checkbox('Usar pesos macroeconômicos', value=True)


# Utilize o valor selecionado na otimização e filtragem de ativos
ativos_validos = filtrar_ativos_validos(carteira, cenario, macro, usar_pesos_macroeconomicos=usar_pesos_macroeconomicos)


if st.button("Gerar Alocação Otimizada"):
    ativos_validos = filtrar_ativos_validos(carteira, cenario, macro)

    if not ativos_validos:
        st.warning("Nenhum ativo com preço atual abaixo do preço-alvo dos analistas.")
    else:
        tickers_validos = [a['ticker'] for a in ativos_validos]
        try:
            if usar_hrp:
                pesos = otimizar_carteira_hrp(tickers_validos)
            else:
                pesos = otimizar_carteira_sharpe(tickers_validos)

            if pesos is not None:
                tickers_completos = set(carteira)
                tickers_usados = set(tickers_validos)
                tickers_zerados = tickers_completos - tickers_usados

                if tickers_zerados:
                    st.subheader("📉 Ativos da carteira atual sem recomendação de aporte")
                    st.write(", ".join(tickers_zerados))

                df_resultado = pd.DataFrame(ativos_validos)
                df_resultado["Alocação (%)"] = (pesos * 100).round(2)
                df_resultado["Valor Alocado (R$)"] = (pesos * aporte).round(2)
                # Calcula valor alocado bruto
                df_resultado["Valor Alocado Bruto (R$)"] = (pesos * aporte)
                
                # Calcula quantidade inteira de ações possível
                df_resultado["Qtd. Ações"] = (df_resultado["Valor Alocado Bruto (R$)"] / df_resultado["preco_atual"]).apply(np.floor)
                
                # Corrige o valor alocado para refletir a quantidade inteira de ações
                df_resultado["Valor Alocado (R$)"] = (df_resultado["Qtd. Ações"] * df_resultado["preco_atual"]).round(2)

                # Cálculo de novos pesos considerando carteira anterior + novo aporte
                # Filtra pesos atuais apenas para os ativos que estão na recomendação
                tickers_resultado = df_resultado["ticker"].tolist()

# Cria um dicionário de ticker -> peso original
                pesos_dict = dict(zip(carteira, pesos_atuais))

# Extrai os pesos apenas para os tickers selecionados
                pesos_atuais_filtrados = np.array([pesos_dict[t] for t in tickers_resultado])

# Continua o cálculo
                valores_atuais = pesos_atuais_filtrados * 1000000  # exemplo: carteira anterior de 1 milhão

                valores_aporte = pesos * aporte
                valores_totais = valores_atuais + valores_aporte
                pesos_finais = valores_totais / valores_totais.sum()

                df_resultado["% na Carteira Final"] = (pesos_finais * 100).round(2)

                st.subheader("📈 Ativos Recomendados para Novo Aporte")
                st.dataframe(df_resultado[["ticker", "setor", "preco_atual", "preco_alvo", "score", "Qtd. Ações", "Valor Alocado (R$)", "% na Carteira Final"]])
                # Calcular o valor total utilizado no aporte
                valor_utilizado = df_resultado["Valor Alocado (R$)"].sum()
                troco = aporte - valor_utilizado
                
                # Mostrar o troco abaixo da tabela
                valor_utilizado = df_resultado["Valor Alocado (R$)"].sum()
                troco = aporte - valor_utilizado
                
                st.markdown(f"**💵 Troco (valor restante do aporte): R$ {troco:,.2f}**")




            else:
                st.error("Falha na otimização da carteira.")
        except Exception as e:
            st.error(f"Erro na otimização: {str(e)}")
            
with st.expander("ℹ️ Como funciona a sugestão"):
    st.markdown("""
    - O cenário macroeconômico é classificado automaticamente com base em **Selic** e **IPCA**.
    - São priorizados ativos com **preço atual abaixo do preço-alvo dos analistas**.
    - Ativos de **setores favorecidos pelo cenário atual** recebem um bônus no score.
    - Exportadoras ganham bônus adicionais com **dólar alto** ou **petróleo acima de US$ 80**.
    - O método de otimização pode ser:
        - **Sharpe máximo** (baseado na relação risco/retorno histórica).
        - **HRP** (Hierarchical Risk Parity), que diversifica riscos sem estimar retornos.
    """)

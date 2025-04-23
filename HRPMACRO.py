import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime
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
    'SAPR11.SA': 'Utilidades Públicas',
    'SAPR3.SA': 'Utilidades Públicas',
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



# ========= MACRO ==========

# Funções para obter dados do BCB

def buscar_projecoes_focus(indicador, ano=datetime.datetime.now().year):
    indicador_map = {
        "IPCA": "IPCA",
        "Selic": "Selic",
        "PIB Total": "PIB Total",
        "Câmbio": "Câmbio"
    }
    
    nome_indicador = indicador_map.get(indicador)
    if not nome_indicador:
        return None

    base_url = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/"
    url = f"{base_url}ExpectativasMercadoTop5Anuais?$top=100000&$filter=Indicador eq '{nome_indicador}'&$format=json&$select=Indicador,Data,DataReferencia,Mediana"

    try:
        response = requests.get(url)
        response.raise_for_status()
        dados = response.json()["value"]

        df = pd.DataFrame(dados)
        df = df[df["DataReferencia"].str.contains(str(ano))]
        df = df.sort_values("Data", ascending=False)

        if df.empty:
            raise ValueError(f"Nenhum dado encontrado para {indicador} em {ano}.")

        return float(df.iloc[0]["Mediana"])
    
    except Exception as e:
        print(f"Erro ao buscar {indicador} no Boletim Focus: {e}")
        return None


def obter_macro():
    macro = {
        "ipca": buscar_projecoes_focus("IPCA"),
        "selic": buscar_projecoes_focus("Selic"),
        "pib": buscar_projecoes_focus("PIB Total"),
        "cambio": buscar_projecoes_focus("Câmbio")
    }

    # Preços de commodities
    macro["soja"] = obter_preco_commodity("ZS=F", nome="Soja")
    macro["milho"] = obter_preco_commodity("ZC=F", nome="Milho")
    macro["minerio"] = obter_preco_commodity("BZ=F", nome="Minério de Ferro")

    return macro


# Função genérica para obter preços via yfinance

def obter_preco_yf(ticker, nome="Ativo"):
    try:
        dados = yf.Ticker(ticker).history(period="5d")
        if not dados.empty and 'Close' in dados.columns:
            return float(dados['Close'].dropna().iloc[-1])
        else:
            st.warning(f"Preço indisponível para {nome}.")
            return None
    except Exception as e:
        st.error(f"Erro ao obter preço de {nome} ({ticker}): {e}")
        return None

def obter_preco_commodity(ticker, nome="Commodity"):
    return obter_preco_yf(ticker, nome)

def obter_preco_petroleo():
    return obter_preco_yf("CL=F", "Petróleo")

# Funções de pontuação individual

def pontuar_selic(selic):
    return max(min(2 - (selic - 9) / 2, 2), -1)  # Quanto menor a Selic, melhor (até 9%)


def pontuar_ipca(ipca):
    return max(min(2 - (ipca - 3) / 2, 2), -1)  # Ideal abaixo de 3%


def pontuar_dolar(cambio):
    if dolar < 4.8:
        return 1
    elif dolar <= 5.2:
        return 0
    else:
        return -1

def pontuar_pib(pib):
    return max(min((pib - 0.5), 2), -1)  # PIB acima de 2 é ótimo, abaixo de 0 ruim


def pontuar_soja_milho(preco_soja, preco_milho):
    if preco_soja is not None and preco_milho is not None:
        if preco_soja > 12 and preco_milho > 5:
            return 2
        elif preco_soja > 11 and preco_milho > 4:
            return 1
        elif preco_soja > 10 and preco_milho > 3:
            return 0
        else:
            return -1
    return 0

def pontuar_minerio(preco_minerio):
    if preco_minerio is not None:
        if preco_minerio > 120:
            return 2
        elif preco_minerio > 100:
            return 1
        elif preco_minerio > 80:
            return 0
        else:
            return -1
    return 0

def pontuar_macro(m):
    score = {}
    score["juros"] = pontuar_selic(m["selic"]) if m.get("selic") else 0
    score["inflação"] = pontuar_ipca(m["ipca"]) if m.get("ipca") else 0
    score["cambio"] = pontuar_dolar(m["cambio"]) if m.get("cambio") else 0
    score["pib"] = pontuar_pib(m["pib"]) if m.get("pib") else 0
    score["commodities_agro"] = pontuar_soja_milho(m.get("soja"), m.get("milho"))
    score["commodities_minerio"] = pontuar_minerio(m.get("minerio"))
    return score



# Funções para preço-alvo e preço atual

def obter_preco_alvo(ticker):
    try:
        return yf.Ticker(ticker).info.get('targetMeanPrice', None)
    except Exception as e:
        st.warning(f"Erro ao obter preço-alvo de {ticker}: {e}")
        return None

def obter_preco_atual(ticker):
    try:
        dados = yf.Ticker(ticker).history(period="1d")
        if not dados.empty:
            return dados['Close'].iloc[-1]
    except Exception as e:
        st.warning(f"Erro ao obter preço atual de {ticker}: {e}")
    return None


def classificar_cenario_macro(score_dict):
    total = sum(score_dict.values())
    if total >= 7:
        return "Expansão Forte"
    elif total >= 4:
        return "Expansão Moderada"
    elif total >= 1:
        return "Estável"
    elif total >= -2:
        return "Contração Moderada"
    else:
        return "Contração Forte"



#===========PESOS FALTANTES======
def completar_pesos(tickers_originais, pesos_calculados):
    """
    Garante que todos os ativos originais estejam presentes nos pesos finais,
    atribuindo 0 para os que foram excluídos na otimização.
    """
    pesos_completos = pd.Series(0.0, index=tickers_originais)
    for ticker in pesos_calculados.index:
        pesos_completos[ticker] = pesos_calculados[ticker]
    return pesos_completos

        

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



def filtrar_ativos_validos(carteira, setores_por_ticker, setores_por_cenario, macro, calcular_score):
    score_macro = pontuar_macro(macro)
    cenario = classificar_cenario(score_macro)
    setores_favorecidos = setores_por_cenario.get(cenario, [])

    ativos_validos = []
    for ticker in carteira:
        setor = setores_por_ticker.get(ticker, None)
        preco_atual = obter_preco_atual(ticker)
        preco_alvo = obter_preco_alvo(ticker)

        if preco_atual is None or preco_alvo is None:
            continue

        favorecido = setor in setores_favorecidos
        score = calcular_score(preco_atual, preco_alvo, favorecido, ticker, macro)

        ativos_validos.append({
            "ticker": ticker,
            "setor": setor,
            "cenario": cenario,
            "preco_atual": preco_atual,
            "preco_alvo": preco_alvo,
            "score": score,
            "favorecido": favorecido
        })

    ativos_validos.sort(key=lambda x: x['score'], reverse=True)
    return ativos_validos

# ========= OTIMIZAÇÃO ==========



def obter_preco_diario_ajustado(tickers):
    dados_brutos = yf.download(tickers, period="7y", auto_adjust=False)

    # Forçar tickers a ser lista, mesmo se for string
    if isinstance(tickers, str):
        tickers = [tickers]

    if isinstance(dados_brutos.columns, pd.MultiIndex):
        if 'Adj Close' in dados_brutos.columns.get_level_values(0):
            return dados_brutos['Adj Close']
        elif 'Close' in dados_brutos.columns.get_level_values(0):
            return dados_brutos['Close']
        else:
            raise ValueError("Colunas 'Adj Close' ou 'Close' não encontradas nos dados.")
    else:
        # Apenas um ticker e colunas simples
        if 'Adj Close' in dados_brutos.columns:
            return dados_brutos[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
        elif 'Close' in dados_brutos.columns:
            return dados_brutos[['Close']].rename(columns={'Close': tickers[0]})
        else:
            raise ValueError("Coluna 'Adj Close' ou 'Close' não encontrada nos dados.")


def otimizar_carteira_sharpe(tickers, carteira_atual, taxa_risco_livre=0.0001):
    """
    Otimiza a carteira com base no índice de Sharpe, com melhorias de robustez e controle de concentração.
    """
    dados = obter_preco_diario_ajustado(tickers)
    dados = dados.ffill().bfill()  # Preenche valores faltantes

    # Retornos logarítmicos
    retornos = np.log(dados / dados.shift(1)).dropna()
    tickers_validos = retornos.columns.tolist()
    n = len(tickers_validos)

    if n == 0:
        st.error("Nenhum dado de retorno válido disponível para os ativos selecionados.")
        return pd.Series(0.0, index=tickers)

    media_retorno = retornos.mean()
    cov_matrix = LedoitWolf().fit(retornos).covariance_
    cov = pd.DataFrame(cov_matrix, index=retornos.columns, columns=retornos.columns)

    def sharpe_neg(pesos):
        retorno_esperado = np.dot(pesos, media_retorno) - taxa_risco_livre
        volatilidade = np.sqrt(pesos @ cov.values @ pesos.T)
        return -retorno_esperado / volatilidade if volatilidade != 0 else 0

    # Pesos iniciais baseados na carteira atual
    pesos_iniciais = np.array([carteira_atual.get(t, 0.0) for t in tickers_validos])
    pesos_iniciais = pesos_iniciais / pesos_iniciais.sum() if pesos_iniciais.sum() > 0 else np.ones(n) / n

    # Limites por ativo: mínimo de 1%, máximo de 20%
    limites = [(0.01, 0.20) for _ in range(n)]

    # Restrição: soma dos pesos = 1
    restricoes = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # Otimização com método robusto
    resultado = minimize(
        sharpe_neg,
        pesos_iniciais,
        method='SLSQP',
        bounds=limites,
        constraints=restricoes,
        options={'disp': False, 'maxiter': 1000}
    )

    if resultado.success and not np.isnan(resultado.fun):
        pesos_otimizados = pd.Series(resultado.x, index=tickers_validos)
        return completar_pesos(tickers, pesos_otimizados)
    else:
        st.warning("Otimização falhou ou retornou valor inválido. Usando pesos uniformes.")
        pesos_uniformes = pd.Series(np.ones(n) / n, index=tickers_validos)
        return completar_pesos(tickers, pesos_uniformes)




def otimizar_carteira_hrp(tickers, carteira_atual):
    """
    Otimiza a carteira com HRP, ajustando os pesos finais com base nos ativos válidos.
    """
    dados = obter_preco_diario_ajustado(tickers)
    dados = dados.dropna(axis=1, how='any')
    tickers_validos = dados.columns.tolist()

    if len(tickers_validos) < 2:
        st.error("Número insuficiente de ativos com dados válidos para otimização.")
        return pd.Series(0.0, index=tickers)

    retornos = dados.pct_change().dropna()
    correlacao = retornos.corr()
    dist = np.sqrt((1 - correlacao) / 2)

    dist_condensada = squareform(dist.values, checks=False)
    linkage_matrix = linkage(dist_condensada, method='single')

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
            df1 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df1])
            sort_ix = sort_ix.sort_index()
        return sort_ix.tolist()

    def get_recursive_bisection(cov, sort_ix):
        w = pd.Series(1, index=sort_ix)
        cluster_items = [sort_ix]

        while len(cluster_items) > 0:
            cluster_items = [i[j:k] for i in cluster_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for cluster in cluster_items:
                c_items = cluster
                c_var = cov.loc[c_items, c_items].values
                inv_diag = 1. / np.diag(c_var)
                parity_w = inv_diag / inv_diag.sum()
                alloc = parity_w.sum()
                w[c_items] *= parity_w * alloc
        return w / w.sum()

    cov_matrix = LedoitWolf().fit(retornos).covariance_
    cov_df = pd.DataFrame(cov_matrix, index=retornos.columns, columns=retornos.columns)
    sort_ix = get_quasi_diag(linkage_matrix)
    ordered_tickers = [retornos.columns[i] for i in sort_ix]
    pesos_hrp = get_recursive_bisection(cov_df, ordered_tickers)

    return completar_pesos(tickers, pesos_hrp)






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


# --- SIDEBAR ---
with st.sidebar:
    st.header("Parâmetros")
    st.markdown("### Dados dos Ativos")

    # Tickers e pesos default
    tickers_default = [
        "AGRO3.SA", "BBAS3.SA", "BBSE3.SA", "BPAC11.SA", "EGIE3.SA",
        "ITUB4.SA", "PRIO3.SA", "PSSA3.SA", "SAPR11.SA", "SBSP3.SA",
        "VIVT3.SA", "WEGE3.SA", "TOTS3.SA", "B3SA3.SA", "TAEE11.SA"
    ]
    pesos_default = [
        0.07, 0.06, 0.07, 0.07, 0.08,
        0.07, 0.12, 0.09, 0.06, 0.04,
        0.1, 0.18, 0.04, 0.01, 0.2
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
                pesos = otimizar_carteira_hrp(tickers_validos, carteira)

            else:
                pesos = otimizar_carteira_sharpe(tickers_validos, carteira)

            if pesos is not None:
                tickers_completos = set(carteira)
                tickers_usados = set(tickers_validos)
                tickers_zerados = tickers_completos - tickers_usados

                
                # Ativos da carteira atual sem recomendação
            if tickers_zerados:
                st.subheader("📉 Ativos da carteira atual sem recomendação de aporte")
                st.write(", ".join(tickers_zerados))
            
            # Cria DataFrame com todos os tickers da carteira original
            todos_os_tickers = list(carteira.keys())
            df_resultado_completo = pd.DataFrame({'ticker': todos_os_tickers})
            
            # Junta com os dados dos ativos válidos (os que passaram nos filtros)
            df_validos = pd.DataFrame(ativos_validos)
            df_resultado = df_resultado_completo.merge(df_validos, on='ticker', how='left')
            
            # Preenche colunas faltantes para os ativos zerados
            df_resultado["preco_atual"] = df_resultado["preco_atual"].fillna(0)
            df_resultado["preco_alvo"] = df_resultado["preco_alvo"].fillna(0)
            df_resultado["score"] = df_resultado["score"].fillna(0)
            df_resultado["setor"] = df_resultado["setor"].fillna("Não recomendado")
            
            # Mapeia os pesos calculados para os tickers (os ausentes recebem 0)
            pesos_dict = dict(zip(tickers_validos, pesos))
            df_resultado["peso_otimizado"] = df_resultado["ticker"].map(pesos_dict).fillna(0)
            
            # Calcula valor alocado bruto e quantidade de ações
            df_resultado["Valor Alocado Bruto (R$)"] = df_resultado["peso_otimizado"] * aporte
            df_resultado["Qtd. Ações"] = (df_resultado["Valor Alocado Bruto (R$)"] / df_resultado["preco_atual"])\
                .replace([np.inf, -np.inf], 0).fillna(0).apply(np.floor)
            df_resultado["Valor Alocado (R$)"] = (df_resultado["Qtd. Ações"] * df_resultado["preco_atual"]).round(2)
            
            # Cálculo de novos pesos considerando carteira anterior + novo aporte
            tickers_resultado = df_resultado["ticker"].tolist()
            pesos_atuais_dict = dict(zip(carteira, pesos_atuais))
            pesos_atuais_filtrados = np.array([pesos_atuais_dict[t] for t in tickers_resultado])
            valores_atuais = pesos_atuais_filtrados * 1_000_000  # exemplo: carteira anterior de 1 milhão
            
            valores_aporte = df_resultado["Valor Alocado (R$)"].to_numpy()
            valores_totais = valores_atuais + valores_aporte
            pesos_finais = valores_totais / valores_totais.sum()
            
            df_resultado["% na Carteira Final"] = (pesos_finais * 100).round(2)
            
            # Exibe a tabela final
            st.subheader("📈 Ativos Recomendados para Novo Aporte")
            st.dataframe(df_resultado[[
                "ticker", "setor", "preco_atual", "preco_alvo", "score", "Qtd. Ações",
                "Valor Alocado (R$)", "% na Carteira Final"
            ]])
            
            # Mostra pesos da HRP como comparação
            pesos_hrp = otimizar_carteira_hrp(todos_os_tickers, carteira)
            st.dataframe(pesos_hrp.rename("Peso HRP"))
            
            # Troco do aporte
            valor_utilizado = df_resultado["Valor Alocado (R$)"].sum()
            troco = aporte - valor_utilizado
            st.markdown(f"💰 **Valor utilizado no aporte:** R$ {valor_utilizado:,.2f}")
            st.markdown(f"🔁 **Troco (não alocado):** R$ {troco:,.2f}")


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

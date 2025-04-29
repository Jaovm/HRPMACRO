import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.optimize import minimize

st.set_page_config(page_title="Sugestão de Carteira", layout="wide")

def get_bcb_hist(code, start, end):
    """Baixa série histórica mensal do BCB para um código SGS."""
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json&dataInicial={start}&dataFinal={end}"
    r = requests.get(url)
    if r.status_code == 200:
        df = pd.DataFrame(r.json())
        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        df['valor'] = df['valor'].str.replace(",", ".").astype(float)
        return df.set_index('data')['valor']
    else:
        return pd.Series(dtype=float)

def obter_preco_petroleo_hist(start, end):
    """Baixa preço histórico mensal do petróleo Brent (BZ=F) do Yahoo Finance."""
    df = yf.download("BZ=F", start=start, end=end, interval="1mo", progress=False)
    if not df.empty:
        df.index = pd.to_datetime(df.index)
        return df['Close']
    return pd.Series(dtype=float)

def montar_historico_7anos(tickers, setores_por_ticker, start='2018-01-01'):
    """Gera histórico dos últimos 7 anos (em memória, sem salvar em CSV)."""
    hoje = datetime.date.today()
    inicio = pd.to_datetime(start)
    final = hoje
    datas = pd.date_range(inicio, final, freq='M').normalize()
    
    # Baixar séries macro históricas do BCB
    selic_hist = get_bcb_hist(432, inicio.strftime('%d/%m/%Y'), final.strftime('%d/%m/%Y'))
    ipca_hist = get_bcb_hist(433, inicio.strftime('%d/%m/%Y'), final.strftime('%d/%m/%Y'))
    dolar_hist = get_bcb_hist(1, inicio.strftime('%d/%m/%Y'), final.strftime('%d/%m/%Y'))
    petroleo_hist = obter_preco_petroleo_hist(inicio.strftime('%Y-%m-%d'), final.strftime('%Y-%m-%d'))
    
    # Normalizar todos os índices para garantir compatibilidade
    selic_hist.index = pd.to_datetime(selic_hist.index).normalize()
    ipca_hist.index = pd.to_datetime(ipca_hist.index).normalize()
    dolar_hist.index = pd.to_datetime(dolar_hist.index).normalize()
    petroleo_hist.index = pd.to_datetime(petroleo_hist.index).normalize()
    
    macro_df = pd.DataFrame(index=datas)
    macro_df['selic'] = selic_hist.reindex(datas, method='ffill')
    macro_df['ipca'] = ipca_hist.reindex(datas, method='ffill')
    macro_df['dolar'] = dolar_hist.reindex(datas, method='ffill')
    macro_df['petroleo'] = petroleo_hist.reindex(datas, method='ffill')
    macro_df = macro_df.fillna(method='ffill').fillna(method='bfill')

    historico = []
    for data in datas:
        macro = {
            "ipca": macro_df.loc[data, "ipca"],
            "selic": macro_df.loc[data, "selic"],
            "dolar": macro_df.loc[data, "dolar"],
            "pib": 2,
            "petroleo": macro_df.loc[data, "petroleo"],
            "soja": None,
            "milho": None,
            "minerio": None
        }
        cenario = classificar_cenario_macro(
            ipca=macro["ipca"],
            selic=macro["selic"],
            dolar=macro["dolar"],
            pib=macro["pib"],
            preco_soja=macro["soja"],
            preco_milho=macro["milho"],
            preco_minerio=macro["minerio"],
            preco_petroleo=macro["petroleo"]
        )
        score_macro = pontuar_macro(macro)
        for ticker in tickers:
            setor = setores_por_ticker.get(ticker, None)
            favorecido = calcular_favorecimento_continuo(setor, score_macro)
            historico.append({
                "data": str(data.date()),
                "cenario": cenario,
                "ticker": ticker,
                "setor": setor,
                "favorecido": favorecido
            })
    df_hist = pd.DataFrame(historico)
    return df_hist

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
    'TUPY3.SA': 'Indústria e Bens de Capital',

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
    'TUPY3.SA',
]


# Mapeamento dos setores mais favorecidos em cada fase do ciclo macroeconômico.
# Ajuste conforme mudanças de conjuntura ou inclusão de novos setores.
setores_por_cenario = {
    # Crescimento acelerado, demanda forte e apetite a risco.
    "Expansão Forte": [
        'Consumo Discricionário',  # Ex: varejo, turismo, educação privada
        'Tecnologia',
        'Indústria e Bens de Capital',
        'Agronegócio',
        'Mineração e Siderurgia',
        'Petróleo, Gás e Biocombustíveis'
    ],
    # Crescimento moderado, ainda com bom apetite, mas já com busca por qualidade.
    "Expansão Moderada": [
        'Consumo Discricionário',
        'Tecnologia',
        'Indústria e Bens de Capital',
        'Agronegócio',
        'Mineração e Siderurgia',
        'Petróleo, Gás e Biocombustíveis',
        'Saúde'  # Começa a ganhar tração em cenários menos exuberantes
    ],
    # Economia estável, equilíbrio entre risco e proteção, preferência por setores defensivos.
    "Estável": [
        'Saúde',
        'Bancos',
        'Seguradoras',
        'Bolsas e Serviços Financeiros',
        'Consumo Básico',
        'Utilidades Públicas',
        'Comunicação'
    ],
    # Início de desaceleração, foco em proteção e estabilidade de receita.
    "Contração Moderada": [
        'Bancos',
        'Seguradoras',
        'Consumo Básico',
        'Utilidades Públicas',
        'Saúde',
        'Energia Elétrica',
        'Comunicação'
    ],
    # Contração severa, recessão; apenas setores mais resilientes.
    "Contração Forte": [
        'Utilidades Públicas',
        'Consumo Básico',
        'Energia Elétrica',
        'Saúde'
    ]
}

# DICA: Para obter todos os setores únicos usados em qualquer cenário:
todos_setores = set()
for setores in setores_por_cenario.values():
    todos_setores.update(setores)
# todos_setores agora contém todos os setores possíveis


# ========= MACRO ==========

# Funções para obter dados do BCB

@st.cache_data(ttl=86400)
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
        "petroleo": obter_preco_petroleo(),
        "dolar": buscar_projecoes_focus("Câmbio"),
        "soja": obter_preco_commodity("ZS=F", nome="Soja"),
        "milho": obter_preco_commodity("ZC=F", nome="Milho"),
        "minerio": obter_preco_commodity("BZ=F", nome="Minério de Ferro")
    }
    return macro

@st.cache_data(ttl=86400)
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

@st.cache_data(ttl=86400)
def obter_preco_commodity(ticker, nome="Commodity"):
    return obter_preco_yf(ticker, nome)

@st.cache_data(ttl=86400)
def obter_preco_petroleo():
    return obter_preco_yf("BZ=F", "Petróleo")

# Funções de pontuação individual

PARAMS = {
    "selic_neutra": 7.0,
    "ipca_meta": 3,
    "ipca_tolerancia": 1.5,
    "dolar_ideal": 5.30,
}

# ==================== FUNÇÕES DE PONTUAÇÃO ====================

def pontuar_ipca(ipca):
    if ipca is None or pd.isna(ipca):
        return 0
    meta = PARAMS["ipca_meta"]
    tolerancia = PARAMS["ipca_tolerancia"]
    # Dentro da meta: 10 pontos
    if meta - tolerancia <= ipca <= meta + tolerancia:
        return 10
    # Até 1% acima da tolerância: 5 pontos
    elif ipca <= meta + tolerancia + 1:
        return 5
    # Muito acima do teto da meta: penalização pesada
    elif ipca > meta + tolerancia + 1:
        return 0
    # Abaixo da banda: 3 pontos (deflação)
    else:
        return 3

def pontuar_selic(selic):
    if selic is None or pd.isna(selic):
        return 0
    neutra = PARAMS["selic_neutra"]
    # Dentro da neutralidade: 10 pontos
    if abs(selic - neutra) <= 0.5:
        return 10
    # Até 2% acima: 4 pontos
    elif selic > neutra and selic <= neutra + 2:
        return 4
    # Muito acima da neutra: 0 pontos
    elif selic > neutra + 2:
        return 0
    # Abaixo da neutra: 6 pontos (ainda expansionista)
    else:
        return 6


def pontuar_dolar(dolar):
    if dolar is None or pd.isna(dolar):
        return 0
    ideal = PARAMS["dolar_ideal"]
    desvio = abs(dolar - ideal)
    return max(0, 10 - desvio * 2)

def pontuar_pib(pib):
    if pib is None or pd.isna(pib):
        return 0
    ideal = 2.0
    if pib >= ideal:
        return min(10, 8 + (pib - ideal) * 2)
    else:
        return max(0, 8 - (ideal - pib) * 3)

# Atualize as funções de preço ideal para usar médias móveis

@st.cache_data(ttl=86400)
def calcular_media_movel(ticker, periodo="12mo", intervalo="1mo"):
    """
    Calcula a média móvel do preço de um ativo (ex.: soja, milho, petróleo, minério).
    Retorna float (valor escalar).
    """
    try:
        dados = yf.download(ticker, period=periodo, interval=intervalo, progress=False)
        if not dados.empty:
            media_movel = float(dados['Close'].mean())
            return media_movel
        else:
            st.warning(f"Dados históricos indisponíveis para {ticker}.")
            return None
    except Exception as e:
        st.error(f"Erro ao calcular média móvel para {ticker}: {e}")
        return None

# --- Função para obter preços ideais dinâmicos usando médias móveis ---
def obter_precos_ideais():
    return {
        "soja_ideal": calcular_media_movel("ZS=F", periodo="12mo", intervalo="1mo"),    # Soja
        "milho_ideal": calcular_media_movel("ZC=F", periodo="12mo", intervalo="1mo"),   # Milho
        "minerio_ideal": calcular_media_movel("TIO=F", periodo="12mo", intervalo="1mo"), # Minério de ferro (use o ticker correto para o seu caso)
        "petroleo_ideal": calcular_media_movel("BZ=F", periodo="12mo", intervalo="1mo") # Petróleo Brent
    }

# --- Atualize os parâmetros globais de commodities ---
def atualizar_parametros_com_medias_moveis():
    precos_ideais = obter_precos_ideais()
    PARAMS.update(precos_ideais)
    return PARAMS

# --- Garanta que as funções de pontuação aceitem apenas valores escalares ---
def pontuar_soja(soja):
    if soja is None or pd.isna(soja):
        return 0
    if isinstance(soja, pd.Series):
        soja = float(soja.iloc[0])
    ideal = PARAMS.get("soja_ideal", 13.0)
    if ideal is None or pd.isna(ideal):
        return 0
    desvio = abs(soja - ideal)
    return max(0, 10 - desvio * 1.5)

def pontuar_milho(milho):
    if milho is None or pd.isna(milho):
        return 0
    if isinstance(milho, pd.Series):
        milho = float(milho.iloc[0])
    ideal = PARAMS.get("milho_ideal", 5.5)
    if ideal is None or pd.isna(ideal):
        return 0
    desvio = abs(milho - ideal)
    return max(0, 10 - desvio * 2)

def pontuar_soja_milho(soja, milho):
    """Pontua a média entre soja e milho, para commodities agro."""
    return (pontuar_soja(soja) + pontuar_milho(milho)) / 2
    
def pontuar_minerio(minerio):
    if minerio is None or pd.isna(minerio):
        return 0
    if isinstance(minerio, pd.Series):
        minerio = float(minerio.iloc[0])
    ideal = PARAMS["minerio_ideal"]
    if ideal is None or pd.isna(ideal):
        return 0
    desvio = abs(minerio - ideal)
    return max(0, 10 - desvio * 0.1)

def pontuar_petroleo(petroleo):
    if petroleo is None or pd.isna(petroleo):
        return 0
    if isinstance(petroleo, pd.Series):
        petroleo = float(petroleo.iloc[0])
    ideal = PARAMS["petroleo_ideal"]
    if ideal is None or pd.isna(ideal):
        return 0
    desvio = abs(petroleo - ideal)
    return max(0, 10 - desvio * 0.2)



# --- Garanta que os parâmetros estejam atualizados antes do uso ---
PARAMS = atualizar_parametros_com_medias_moveis()

# Atualize o Streamlit para mostrar os preços ideais
def validar_macro(macro):
    obrigatorios = ["selic", "ipca", "dolar", "pib", "soja", "milho", "minerio", "petroleo"]
    for k in obrigatorios:
        if k

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime
import os
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.optimize import minimize

def carregar_historico_cenarios(path="historico_cenarios.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame(columns=[
            "data", "cenario", "ticker", "setor", "score", "favorecido"
        ])

def salvar_historico_cenarios(df, path="historico_cenarios.csv"):
    df.to_csv(path, index=False)

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

@st.cache_data(ttl=86400)  # Cache por 1 dia
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


# Função genérica para obter preços via yfinance
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

def pontuar_ipca(ipca):
    meta = 3.0
    tolerancia = 1.5
    if meta - tolerancia <= ipca <= meta + tolerancia:
        return 10
    elif ipca < meta - tolerancia:
        return 7  # inflação baixa (risco de deflação)
    else:
        return max(0, 10 - (ipca - (meta + tolerancia)) * 2)

# Função para pontuar a Selic
def pontuar_selic(selic):
    neutra = 9.0
    if selic == neutra:
        return 10
    elif selic < neutra:
        return max(5, 10 - (neutra - selic) * 1.5)  # juros estimulativos
    else:
        return max(0, 10 - (selic - neutra) * 1.5)  # juros contracionistas

# Função para pontuar o Câmbio
def pontuar_dolar(dolar):
    ideal = 5.90
    desvio = abs(dolar - ideal)
    return max(0, 10 - desvio * 2)

# Função para pontuar o PIB
def pontuar_pib(pib):
    ideal = 2.0
    if pib >= ideal:
        return min(10, 8 + (pib - ideal) * 2)
    else:
        return max(0, 8 - (ideal - pib) * 3)


def pontuar_soja(soja):
    ideal = 13.0  # referência média (US$/bushel)
    desvio = abs(soja - ideal)
    return max(0, 10 - desvio * 1.5)

def pontuar_milho(milho):
    ideal = 5.5  # referência média (US$/bushel)
    desvio = abs(milho - ideal)
    return max(0, 10 - desvio * 2)

def pontuar_minerio(minerio):
    ideal = 110.0  # referência média (US$/tonelada)
    desvio = abs(minerio - ideal)
    return max(0, 10 - desvio * 0.1)

def pontuar_petroleo(petroleo):
    ideal = 85.0  # referência média (US$/barril)
    desvio = abs(petroleo - ideal)
    return max(0, 10 - desvio * 0.2)


def pontuar_macro(m):
    score = {}
    score["juros"] = pontuar_selic(m.get("selic"))
    score["inflação"] = pontuar_ipca(m.get("ipca"))
    score["dolar"] = pontuar_dolar(m.get("dolar"))
    score["pib"] = pontuar_pib(m.get("pib"))
    score["commodities_agro"] = pontuar_soja_milho(m.get("soja"), m.get("milho"))
    score["commodities_minerio"] = pontuar_minerio(m.get("minerio"))
    score["commodities_petroleo"] = pontuar_petroleo(m.get("petroleo"))
    return score


def pontuar_soja_milho(soja, milho):
    return (pontuar_soja(soja) + pontuar_milho(milho)) / 2


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

def gerar_ranking_acoes(carteira, macro, usar_pesos_macro=True):
    score_macro = pontuar_macro(macro)
    resultados = []

    for ticker in carteira.keys():
        setor = setores_por_ticker.get(ticker)
        if setor is None:
            st.warning(f"Setor não encontrado para {ticker}. Ignorando.")
            continue

        preco_atual = obter_preco_atual(ticker)
        preco_alvo = obter_preco_alvo(ticker)

        if preco_atual is None or preco_alvo is None or preco_atual == 0:
            st.warning(f"Dados insuficientes para {ticker}. Ignorando.")
            continue

        favorecimento_score = calcular_favorecimento_continuo(setor, score_macro)
        score, detalhe = calcular_score(preco_atual, preco_alvo, favorecimento_score, ticker, setor, macro, usar_pesos_macro, return_details=True)

        resultados.append({
            "ticker": ticker,
            "setor": setor,
            "preço atual": preco_atual,
            "preço alvo": preco_alvo,
            "favorecimento macro": favorecimento_score,
            "score": score,
            "detalhe": detalhe
        })

    df = pd.DataFrame(resultados).sort_values(by="score", ascending=False)

    # Garantir exibição mesmo se algumas colunas estiverem ausentes
    colunas_desejadas = ["ticker", "setor", "preço atual", "preço alvo", "favorecimento macro", "score"]
    colunas_existentes = [col for col in colunas_desejadas if col in df.columns]
    st.dataframe(df[colunas_existentes], use_container_width=True)

    
    with st.expander("🔍 Ver detalhes dos scores"):
        st.dataframe(df[["ticker", "detalhe"]], use_container_width=True)
    
    return df

     

def calcular_score(preco_atual, preco_alvo, favorecimento_score, ticker, setor, macro, usar_pesos_macroeconomicos=True, return_details=False):
    import numpy as np

    if preco_atual == 0:
        return -float("inf"), "Preço atual igual a zero"

    upside = (preco_alvo - preco_atual) / preco_atual
    base_score = upside * 10

    score_macro = 0
    if setor in sensibilidade_setorial and usar_pesos_macroeconomicos:
        s = sensibilidade_setorial[setor]
        score_indicadores = pontuar_macro(macro)  # já normalizado
        for indicador, peso in s.items():
            score_macro += peso * score_indicadores.get(indicador, 0)

    score_macro = np.clip(score_macro, -10, 10)

    bonus = 0
    if ticker in empresas_exportadoras:
        if macro.get('dolar') and macro['dolar'] > 5:
            bonus += 0.05
        if macro.get('petroleo') and macro['petroleo'] > 80:
            bonus += 0.05
    bonus = np.clip(bonus, 0, 0.1)

    score_total = base_score + (0.05 * score_macro) + bonus + (favorecimento_score * 1.5 if usar_pesos_macroeconomicos else 0)
    detalhe = f"upside={upside:.2f}, base={base_score:.2f}, macro={score_macro:.2f}, bonus={bonus:.2f}, favorecimento={favorecimento_score:.2f}"

    return (score_total, detalhe) if return_details else score_total




def classificar_cenario_macro(ipca, selic, dolar, pib,
                              preco_soja=None, preco_milho=None,
                              preco_minerio=None, preco_petroleo=None):

    score_ipca = pontuar_ipca(ipca)
    score_selic = pontuar_selic(selic)
    score_dolar = pontuar_dolar(dolar)
    score_pib = pontuar_pib(pib)

    total_score = score_ipca + score_selic + score_dolar + score_pib

    # Adiciona pontuação de commodities, se fornecidas
    if preco_soja is not None:
        total_score += pontuar_soja(preco_soja)
    if preco_milho is not None:
        total_score += pontuar_milho(preco_milho)
    if preco_minerio is not None:
        total_score += pontuar_minerio(preco_minerio)
    if preco_petroleo is not None:
        total_score += pontuar_petroleo(preco_petroleo)

    # Ajusta escala de classificação
    if total_score >= 60:
        return "Expansão Econômica"
    elif total_score >= 45:
        return "Neutro"
    elif total_score >= 30:
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
    'Bancos':                          {'juros': 1,  'inflação': 0,  'dolar': 0,  'pib': 1,  'commodities_agro': 1, 'commodities_minerio': 1},
    'Seguradoras':                     {'juros': 2,  'inflação': 0,  'dolar': 0,  'pib': 1,  'commodities_agro': 0, 'commodities_minerio': 0},
    'Bolsas e Serviços Financeiros':  {'juros': 1,  'inflação': 0,  'dolar': 0,  'pib': 2,  'commodities_agro': 0, 'commodities_minerio': 0},
    'Energia Elétrica':               {'juros': 2,  'inflação': 1,  'dolar': -1, 'pib': -1, 'commodities_agro': -1, 'commodities_minerio': -1},
    'Petróleo, Gás e Biocombustíveis':{'juros': 0,  'inflação': 0,  'dolar': 2,  'pib': 1,  'commodities_agro': 0,  'commodities_minerio': 0},
    'Mineração e Siderurgia':         {'juros': 0,  'inflação': 0,  'dolar': 2,  'pib': 1,  'commodities_agro': 0,  'commodities_minerio': 2},
    'Indústria e Bens de Capital':    {'juros': -1, 'inflação': -1, 'dolar': -1, 'pib': 2,  'commodities_agro': 0,  'commodities_minerio': 0},
    'Agronegócio':                    {'juros': 0,  'inflação': -1, 'dolar': 2,  'pib': 1,  'commodities_agro': 2,  'commodities_minerio': 0},
    'Saúde':                          {'juros': 0,  'inflação': 0,  'dolar': 0,  'pib': 1,  'commodities_agro': 0,  'commodities_minerio': 0},
    'Tecnologia':                     {'juros': -2, 'inflação': 0,  'dolar': 0,  'pib': 2,  'commodities_agro': -1, 'commodities_minerio': -1},
    'Consumo Discricionário':         {'juros': -2, 'inflação': -1, 'dolar': -1, 'pib': 2,  'commodities_agro': -1, 'commodities_minerio': -1},
    'Consumo Básico':                 {'juros': 1,  'inflação': -2, 'dolar': -1, 'pib': 1,  'commodities_agro': -1, 'commodities_minerio': -1},
    'Comunicação':                    {'juros': 0,  'inflação': 0,  'dolar': -1, 'pib': 1,  'commodities_agro': 0,  'commodities_minerio': 0},
    'Utilidades Públicas':            {'juros': 2,  'inflação': 1,  'dolar': -1, 'pib': -1, 'commodities_agro': -1, 'commodities_minerio': -1}
}

def calcular_favorecimento_continuo(setor, score_macro):
    if setor not in sensibilidade_setorial:
        return 0
    sens = sensibilidade_setorial[setor]
    bruto = sum(score_macro.get(k, 0) * peso for k, peso in sens.items())
    return np.tanh(bruto / 5) * 2  # suaviza com tangente hiperbólica


    


def filtrar_ativos_validos(carteira, setores_por_ticker, setores_por_cenario, macro, calcular_score):
    # Extrair valores individuais do dicionário de pontuação
    score_macro = pontuar_macro(macro)
    ipca = score_macro.get("inflação")
    selic = score_macro.get("juros")
    dolar = score_macro.get("dolar")
    pib = score_macro.get("pib")

    # Agora chama a função passando os parâmetros individuais
    cenario = classificar_cenario_macro(ipca, selic, dolar, pib, 
                                        preco_soja=macro.get("soja"), 
                                        preco_milho=macro.get("milho"), 
                                        preco_minerio=macro.get("minerio"), 
                                        preco_petroleo=macro.get("petroleo"))
    
    # Exibir as pontuações e o cenário


    # Obter os setores válidos conforme o cenário
    setores_cidos = setores_por_cenario.get(cenario, [])

    # Inicializar a lista de ativos válidos
    ativos_validos = []
    for ticker in carteira:
        setor = setores_por_ticker.get(ticker, None)
        preco_atual = obter_preco_atual(ticker)
        preco_alvo = obter_preco_alvo(ticker)

        if preco_atual is None or preco_alvo is None:
            continue

        favorecimento_score = calcular_favorecimento_continuo(setor, macro)
        score = calcular_score(preco_atual, preco_alvo, favorecimento_score, ticker, setor, macro, usar_pesos_macroeconomicos=True, return_details=False)

        # Adicionar o ativo à lista de ativos válidos
        ativos_validos.append({
            "ticker": ticker,
            "setor": setor,
            "cenario": cenario,
            "preco_atual": preco_atual,
            "preco_alvo": preco_alvo,
            "score": score,
            "favorecido": favorecimento_score
        })

    # Ordenar os ativos válidos pelo score
    ativos_validos.sort(key=lambda x: x['score'], reverse=True)

    return ativos_validos


# ========= OTIMIZAÇÃO ==========

@st.cache_data(ttl=86400)
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


def otimizar_carteira_sharpe(tickers, carteira_atual, taxa_risco_livre=0.0001, favorecimentos=None):
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
    if favorecimentos:
        fav_array = np.array([max(0, favorecimentos.get(t, 0)) for t in tickers_validos])
        # Pesos iniciais proporcionais ao favorecimento (ou uniformes se tudo zero)
        if fav_array.sum() > 0:
            pesos_iniciais = fav_array / fav_array.sum()
        else:
            pesos_iniciais = np.ones(n) / n
        # Limite máximo 20% + até 10% extra se favorecido (exemplo)
        max_limits = 0.2 + 0.1 * fav_array / (fav_array.max() if fav_array.max() > 0 else 1)
        limites = [(0.01, float(mx)) for mx in max_limits]
        # Opcional: ajustar retorno esperado pelo favorecimento (deixe comentado se não quiser)
        # media_retorno = media_retorno * (1 + 0.5 * fav_array)
    else:
        pesos_iniciais = np.array([carteira_atual.get(t, 0.0) for t in tickers_validos])
        pesos_iniciais = pesos_iniciais / pesos_iniciais.sum() if pesos_iniciais.sum() > 0 else np.ones(n) / n
        limites = [(0.01, 0.20) for _ in range(n)]

    def sharpe_neg(pesos):
        retorno_esperado = np.dot(pesos, media_retorno) - taxa_risco_livre
        volatilidade = np.sqrt(pesos @ cov.values @ pesos.T)
        return -retorno_esperado / volatilidade if volatilidade != 0 else 0

    # Pesos iniciais baseados na carteira atual
    # --- NOVO: usar favorecimento do cenário ---
    if favorecimentos:
        fav_array = np.array([max(0, favorecimentos.get(t, 0)) for t in tickers_validos])
        # Pesos iniciais proporcionais ao favorecimento (ou uniformes se tudo zero)
        if fav_array.sum() > 0:
            pesos_iniciais = fav_array / fav_array.sum()
        else:
            pesos_iniciais = np.ones(n) / n
        # Limite máximo 20% + até 10% extra se favorecido
        max_limits = 0.2 + 0.1 * fav_array / (fav_array.max() if fav_array.max() > 0 else 1)
        limites = [(0.01, float(mx)) for mx in max_limits]
    else:
        pesos_iniciais = np.array([carteira_atual.get(t, 0.0) for t in tickers_validos])
        pesos_iniciais = pesos_iniciais / pesos_iniciais.sum() if pesos_iniciais.sum() > 0 else np.ones(n) / n
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




def otimizar_carteira_hrp(tickers, carteira_atual, favorecimentos=None):
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

        # --- NOVO: ajuste final pelo favorecimento ---
    if favorecimentos:
        fav_array = np.array([1 + max(0, favorecimentos.get(t, 0)) for t in pesos_hrp.index])
        pesos_hrp = pesos_hrp * fav_array
        pesos_hrp = pesos_hrp / pesos_hrp.sum()

    return completar_pesos(tickers, pesos_hrp)

    def montar_historico_sete_anos(tickers, setores_por_ticker, setores_por_cenario):
        historico = []
        hoje = datetime.date.today()
        inicio = hoje - datetime.timedelta(days=365*7)
        datas = pd.date_range(inicio, hoje, freq='M')
        for data in datas:
            macro = obter_macro_em_data(data)  # Você precisa adaptar para pegar macro de cada mês
            cenario = classificar_cenario_macro(
                ipca=macro.get("ipca"),
                selic=macro.get("selic"),
                dolar=macro.get("dolar"),
                pib=macro.get("pib"),
                preco_soja=macro.get("soja"),
                preco_milho=macro.get("milho"),
                preco_minerio=macro.get("minerio"),
                preco_petroleo=macro.get("petroleo"))
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
        df_historico = pd.DataFrame(historico)
        df_historico.to_csv("historico_7anos.csv", index=False)

# ========= STREAMLIT ==========
st.set_page_config(page_title="Sugestão de Carteira", layout="wide")
st.title("📊 Sugestão e Otimização de Carteira: Cenário Projetado")

st.markdown("---")


macro = obter_macro()
cenario = classificar_cenario_macro(
    ipca=macro.get("ipca"),
    selic=macro.get("selic"),
    dolar=macro.get("dolar"),
    pib=macro.get("pib"),
    preco_soja=macro.get("soja"),
    preco_milho=macro.get("milho"),
    preco_minerio=macro.get("minerio"),
    preco_petroleo=macro.get("petroleo")
)

with st.sidebar:
    st.header("Ajuste Manual dos Indicadores Macro")
    macro_manual = {}
    for indicador in ["ipca", "selic", "pib", "dolar"]:
        macro_manual[indicador] = st.number_input(
            f"{indicador.upper()} (ajuste, opcional)", 
            value=macro[indicador] if macro[indicador] else 0.0,
            step=0.01
        )
    usar_macro_manual = st.checkbox("Usar ajustes manuais acima?")
    if usar_macro_manual:
        macro.update(macro_manual)

score_macro = pontuar_macro(macro)
score_medio = round(np.mean(list(score_macro.values())), 2)
st.markdown(f"### 🧭 Cenário Macroeconômico Atual: **{cenario}**")
st.markdown("### 📉 Indicadores Macroeconômicos")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Selic (%)", f"{macro['selic']:.2f}")
col2.metric("IPCA (%)", f"{macro['ipca']:.2f}")
col3.metric("PIB (%)", f"{macro['pib']:.2f}")
col4.metric("Dólar (R$)", f"{macro['dolar']:.2f}")
col5.metric("Petróleo (US$)", f"{macro['petroleo']:.2f}" if macro['petroleo'] else "N/A")


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
        0.1, 0.18, 0.04, 0.01, 0.02
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



# Gerar ranking geral com base no score macro + preço alvo
st.subheader("🏆 Ranking Geral de Ações (com base no score)")
carteira = dict(zip(tickers, pesos_atuais))
ranking_df = gerar_ranking_acoes(carteira, macro, usar_pesos_macro=True)



aporte = st.number_input("💰 Valor do aporte mensal (R$)", min_value=100.0, value=1000.0, step=100.0)
usar_hrp = st.checkbox("Utilizar HRP em vez de Sharpe máximo")




# Utilize o valor selecionado na otimização e filtragem de ativos
ativos_validos = filtrar_ativos_validos(carteira, setores_por_ticker, setores_por_cenario, macro, calcular_score)

    
favorecimentos = {a['ticker']: a['favorecido'] for a in ativos_validos}

if st.button("Gerar Alocação Otimizada"):
    ativos_validos = filtrar_ativos_validos(carteira, setores_por_ticker, setores_por_cenario, macro, calcular_score)
    historico = carregar_historico_cenarios()

    # Monta novo registro para cada ativo válido
    hoje = datetime.datetime.now().strftime("%Y-%m-%d")
    novos_registros = []
    for ativo in ativos_validos:
        novos_registros.append({
            "data": hoje,
            "cenario": cenario,
            "ticker": ativo["ticker"],
            "setor": ativo["setor"],
            "score": ativo["score"],
            "favorecido": ativo["favorecido"]
        })
    
    # Atualiza DataFrame e salva, evitando warning do pandas
    df_novos = pd.DataFrame(novos_registros)
    if not df_novos.empty:
        historico = pd.concat([historico, df_novos], ignore_index=True)
        salvar_historico_cenarios(historico)

    if not ativos_validos:
        st.warning("Nenhum ativo com preço atual abaixo do preço-alvo dos analistas.")
    else:
        tickers_validos = [a['ticker'] for a in ativos_validos]
        try:
            if usar_hrp:
                pesos = otimizar_carteira_hrp(tickers_validos, carteira, favorecimentos=favorecimentos)
            else:
                pesos = otimizar_carteira_sharpe(tickers_validos, carteira, favorecimentos=favorecimentos)

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
            ]], use_container_width=True)

            for i, row in df_resultado.iterrows():
                explicacao = f"O ativo {row['ticker']} foi recomendado porque: "
                explicacao += f"Setor {row['setor']} é favorecido em cenários de {cenario}. "
                if row['favorecido'] > 0:
                    explicacao += "Setor sensível a fatores macro positivos. "
                if row['ticker'] in empresas_exportadoras and macro['dolar'] > 5:
                    explicacao += "Exportadora favorecida por dólar alto. "
                st.info(explicacao)

                st.subheader("🏅 Top 5 empresas que mais se destacaram em cenários similares nos últimos 7 anos")
                
                # Carregue o histórico de 7 anos
                try:
                    historico_7anos = pd.read_csv("historico_7anos.csv")
                except FileNotFoundError:
                    st.info("O histórico dos últimos 7 anos ainda não foi gerado.")
                    historico_7anos = pd.DataFrame()
                
                if historico_7anos.empty:
                    st.info("Sem dados históricos para exibir. Rode a rotina de geração do histórico de 7 anos primeiro.")
                else:
                    # Filtro: cenário igual ao atual e tickers da carteira
                    similares = historico_7anos[(historico_7anos["cenario"] == cenario) &
                                                (historico_7anos["ticker"].isin(carteira.keys()))]
                    if similares.empty:
                        st.info("Nenhum destaque histórico para esse cenário.")
                    else:
                        destaque = (
                            similares.groupby(["ticker", "setor"])
                            .agg(media_favorecido=("favorecido", "mean"),
                                 ocorrencias=("favorecido", "count"))
                            .reset_index()
                            .sort_values(by=["media_favorecido", "ocorrencias"], ascending=False)
                        )
                        st.dataframe(destaque.head(5), use_container_width=True)
                        
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

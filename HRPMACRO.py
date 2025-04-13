import yfinance as yf
import requests
import datetime

def get_selic():
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4189/dados/ultimos/1?formato=json"
    r = requests.get(url).json()
    return float(r[0]['valor'].replace(',', '.'))

def get_ipca():
    url = "https://servicodados.ibge.gov.br/api/v3/agregados/433/dados/ultimos/1"
    r = requests.get(url).json()
    return float(r[0]['resultados'][0]['series'][0]['serie'].values()[0])

def get_pib():
    url = "https://servicodados.ibge.gov.br/api/v3/agregados/5932/periodos/ultimo/variaveis/4099?localidades=N1[all]"
    r = requests.get(url).json()
    valor = list(r[0]['resultados'][0]['series'][0]['serie'].values())[0]
    return float(valor)

def get_cambio():
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados/ultimos/1?formato=json"
    r = requests.get(url).json()
    return float(r[0]['valor'].replace(',', '.'))

def get_petroleo():
    hoje = datetime.datetime.today().strftime('%Y-%m-%d')
    petroleo = yf.download('BZ=F', end=hoje, period='5d')  # Brent
    return round(petroleo['Close'][-1], 2)

def classificar_cenario(selic, ipca, pib):
    if selic > 10 and ipca > 6 and pib < 0.5:
        return "Restritivo"
    elif selic < 7 and ipca < 4 and pib > 1.5:
        return "Expansionista"
    else:
        return "Neutro"

with st.expander("Cenário Macroeconômico Atual"):
    if st.button("Detectar Cenário Atual"):
        with st.spinner("Buscando dados macroeconômicos..."):
            try:
                selic = get_selic()
                ipca = get_ipca()
                pib = get_pib()
                cambio = get_cambio()
                petroleo = get_petroleo()
                cenario = classificar_cenario(selic, ipca, pib)

                st.success(f"Cenário atual: **{cenario}**")
                st.write(f"- SELIC: {selic:.2f}%")
                st.write(f"- IPCA: {ipca:.2f}%")
                st.write(f"- PIB: {pib:.2f}%")
                st.write(f"- Câmbio: R$ {cambio:.2f}")
                st.write(f"- Petróleo (Brent): US$ {petroleo}")

                st.session_state["cenario_atual"] = cenario
            except Exception as e:
                st.error("Erro ao buscar dados: " + str(e))

setores_ativos = {
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


def recomendar_ativos_por_cenario(cenario, setores_ativos, setores_por_cenario):
    setores_favoraveis = setores_por_cenario.get(cenario, [])
    recomendados = [
        ativo for ativo, setor in setores_ativos.items()
        if setor in setores_favoraveis
    ]
    return recomendados, setores_favoraveis

with st.expander("Recomendações baseadas no cenário"):
    if "cenario_atual" in st.session_state:
        cenario = st.session_state["cenario_atual"]
        recomendados, setores_fav = recomendar_ativos_por_cenario(
            cenario, setores_ativos, setores_por_cenario
        )
        st.subheader(f"Setores favorecidos no cenário {cenario}:")
        st.write(", ".join(setores_fav))

        st.subheader("Ações recomendadas da sua carteira:")
        for ativo in recomendados:
            st.markdown(f"- **{ativo}** ({setores_ativos[ativo]})")
    else:
        st.info("Detecte o cenário atual primeiro para obter recomendações.")

# Interface: checkbox para filtrar ativos recomendados
usar_so_recomendados = st.checkbox("Usar apenas ativos recomendados pelo cenário atual", value=True)

# Definição da lista final de ativos
if usar_so_recomendados and "cenario_atual" in st.session_state:
    ativos_para_otimizacao = [
        ativo for ativo, setor in setores_ativos.items()
        if setor in setores_por_cenario[st.session_state["cenario_atual"]]
    ]
else:
    ativos_para_otimizacao = list(setores_ativos.keys())

# Substituir a linha antiga:
# tickers = ["AGRO3.SA", "BBAS3.SA", ...]
# por:
tickers = ativos_para_otimizacao

def calcular_score(ativo, upside, setor, cenario, historico_bom=None):
    score = 0
    pesos = {
        "upside": 0.4,
        "setor_favoravel": 0.3,
        "historico": 0.2,
        "exportadora": 0.1
    }

    score += pesos["upside"] * upside.get(ativo, 0)
    if setor in setores_por_cenario.get(cenario, []):
        score += pesos["setor_favoravel"]
    if historico_bom and ativo in historico_bom:
        score += pesos["historico"]
    if setor == "Exportadoras" and cenario in ["Restritivo", "Neutro"]:  # dólar alto tende a favorecer exportadoras
        score += pesos["exportadora"]

    return round(score, 3)

st.subheader("Pontuação dos ativos")

# Exemplo de dados simulados (você pode puxar real via API)
upside_simulado = {
    "AGRO3.SA": 0.6, "BBAS3.SA": 0.5, "BBSE3.SA": 0.45, "BPAC11.SA": 0.7,
    "EGIE3.SA": 0.4, "ITUB3.SA": 0.55, "PRIO3.SA": 0.75, "PSSA3.SA": 0.35,
    "SAPR3.SA": 0.3, "SBSP3.SA": 0.2, "VIVT3.SA": 0.4, "WEGE3.SA": 0.6,
    "TOTS3.SA": 0.65, "B3SA3.SA": 0.5, "TAEE3.SA": 0.3
}
historico_bom = ["PRIO3.SA", "BBAS3.SA", "WEGE3.SA"]  # Exemplo simples

# Construção da tabela
dados = []
cenario_atual = st.session_state.get("cenario_atual", "Neutro")
for ativo in setores_ativos:
    setor = setores_ativos[ativo]
    score = calcular_score(ativo, upside_simulado, setor, cenario_atual, historico_bom)
    dados.append({"Ativo": ativo, "Setor": setor, "Score": score})

df_score = pd.DataFrame(dados).sort_values(by="Score", ascending=False).reset_index(drop=True)
st.dataframe(df_score)

limite_score = 0.5
ativos_filtrados_score = df_score[df_score["Score"] >= limite_score]["Ativo"].tolist()

tickers = ativos_filtrados_score

st.subheader("Exportar alocação final")

# Criar DataFrame com alocação
df_alocacao = pd.DataFrame({
    "Ativo": list(pesos_finais.keys()),
    "Peso (%)": [round(p * 100, 2) for p in pesos_finais.values()]
})

# Exibir
st.dataframe(df_alocacao)

# Gerar CSV
csv = df_alocacao.to_csv(index=False).encode('utf-8')
st.download_button("📥 Baixar Alocação (CSV)", data=csv, file_name="alocacao_portfolio.csv", mime='text/csv')

# Simular retorno da carteira
retornos_carteira = sum(retornos[ativo] * peso for ativo, peso in pesos_finais.items())
retorno_acumulado_carteira = (1 + retornos_carteira).cumprod()

# IBOV (exemplo com ^BVSP)
precos_ibov = yf.download("^BVSP", start=inicio, end=fim)["Adj Close"]
retornos_ibov = precos_ibov.pct_change().dropna()
retorno_acumulado_ibov = (1 + retornos_ibov).cumprod()

# Unificar série
df_comparativo = pd.DataFrame({
    "Carteira Otimizada": retorno_acumulado_carteira,
    "IBOV": retorno_acumulado_ibov
}).dropna()

# Plotar
st.subheader("📊 Desempenho Carteira vs IBOV")
st.line_chart(df_comparativo)


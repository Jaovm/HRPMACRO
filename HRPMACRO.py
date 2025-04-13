import streamlit as st
import pandas as pd
import numpy as np

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

import streamlit as st
import pandas as pd
import numpy as np

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
# classificar_cenario_macro(macro)
# filtrar_ativos_validos(tickers, cenario, macro)
# otimizar_carteira_sharpe(tickers)
# otimizar_carteira_hrp(tickers)

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

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from HRPMACRO import (
    setores_por_ticker,
    setores_por_cenario,
    obter_preco_diario_ajustado,
    pontuar_macro,
    classificar_cenario_macro,
    calcular_favorecimento_continuo,
    otimizar_carteira_sharpe,
    otimizar_carteira_hrp,
    get_bcb_hist,
)

st.title("Backtest Mensal com Aportes – HRPMACRO (Macroeconômico Histórico, max 30% por ativo)")

valor_aporte = 1000.0
limite_porc_ativo = 0.3
start_date = pd.to_datetime("2018-01-01")
end_date = pd.to_datetime(datetime.date.today())

tickers_str = st.text_input(
    "Tickers elegíveis (ex: PETR4.SA,VALE3.SA,ITUB4.SA)",
    "PETR4.SA,VALE3.SA,ITUB4.SA"
)
tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]

opt_method = st.selectbox(
    "Método de otimização",
    ["Sharpe (macro)", "HRP"],
    index=0
)

if st.button("Executar Backtest Mensal"):
    if not tickers:
        st.warning("Inclua ao menos um ticker.")
        st.stop()

    datas_aporte = pd.date_range(start_date, end_date, freq="MS")
    valor_carteira = []
    datas_carteira = []
    historico_pesos = []
    historico_num_ativos = []
    patrimonio = 0.0

    # Para guardar histórico macroeconômico de cada mês
    historico_macro = []

    # Baixar preços históricos dos ativos e benchmark
    st.write("Baixando preços históricos dos ativos e benchmark (BOVA11.SA)...")
    all_tickers = list(set(tickers + ['BOVA11.SA']))
    precos = obter_preco_diario_ajustado(all_tickers)
    precos = precos.ffill().dropna()

    # Baixar séries macroeconômicas históricas
    st.write("Baixando séries macroeconômicas históricas (Selic, IPCA, Dólar)...")
    selic_hist = get_bcb_hist(432, start_date.strftime('%d/%m/%Y'), end_date.strftime('%d/%m/%Y'))  # Selic
    ipca_hist = get_bcb_hist(433, start_date.strftime('%d/%m/%Y'), end_date.strftime('%d/%m/%Y'))   # IPCA
    dolar_hist = get_bcb_hist(1,    start_date.strftime('%d/%m/%Y'), end_date.strftime('%d/%m/%Y')) # Dólar

    # Montar macro_df mensal alinhado às datas_aporte
    macro_df = pd.DataFrame(index=datas_aporte)
    macro_df['selic'] = selic_hist.reindex(datas_aporte, method='ffill')
    macro_df['ipca'] = ipca_hist.reindex(datas_aporte, method='ffill')
    macro_df['dolar'] = dolar_hist.reindex(datas_aporte, method='ffill')
    macro_df['pib'] = 2.0  # Se quiser, pode estimar/baixar série real de PIB
    macro_df = macro_df.fillna(method='ffill').fillna(method='bfill')

    carteira = {t: 0 for t in tickers}
    caixa = 0.0
    bova11_prices = precos['BOVA11.SA']
    bova11_quantidade = 0
    bova11_patrimonio = []

    for idx, data_aporte in enumerate(datas_aporte):
        # Pega macro daquele mês
        macro = {
            "selic": macro_df.loc[data_aporte, "selic"],
            "ipca": macro_df.loc[data_aporte, "ipca"],
            "dolar": macro_df.loc[data_aporte, "dolar"],
            "pib": macro_df.loc[data_aporte, "pib"],
        }
        historico_macro.append({
            "data": data_aporte,
            "selic": macro["selic"],
            "ipca": macro["ipca"],
            "dolar": macro["dolar"],
            "pib": macro["pib"],
        })

        score_macro = pontuar_macro(macro)
        cenario = classificar_cenario_macro(
            ipca=macro.get("ipca"),
            selic=macro.get("selic"),
            dolar=macro.get("dolar"),
            pib=macro.get("pib"),
        )
        ativos_validos = []
        period_prices = precos.loc[:data_aporte + pd.offsets.MonthEnd(0)].copy()
        for t in tickers:
            setor = setores_por_ticker.get(t)
            if setor is None or t not in period_prices.columns:
                continue
            favorecido = calcular_favorecimento_continuo(setor, score_macro)
            ativos_validos.append(
                {"ticker": t, "setor": setor, "favorecido": favorecido}
            )
        if not ativos_validos:
            valor_carteira.append(patrimonio)
            datas_carteira.append(data_aporte)
            preco_bova = bova11_prices.asof(data_aporte)
            if np.isnan(preco_bova):
                bova11_patrimonio.append(np.nan)
            else:
                qtd_bova = valor_aporte // preco_bova
                bova11_quantidade += qtd_bova
                patrimonio_bova = bova11_quantidade * preco_bova
                bova11_patrimonio.append(patrimonio_bova)
            historico_pesos.append({})
            historico_num_ativos.append(0)
            continue

        ativos_validos_tickers = [a["ticker"] for a in ativos_validos]
        favorecimentos = {a["ticker"]: a["favorecido"] for a in ativos_validos}

        lookback_inicio = data_aporte - pd.DateOffset(months=12)
        lookback_prices = precos.loc[lookback_inicio:data_aporte, ativos_validos_tickers].dropna()
        if len(lookback_prices) < 2:
            valor_carteira.append(patrimonio)
            datas_carteira.append(data_aporte)
            preco_bova = bova11_prices.asof(data_aporte)
            if np.isnan(preco_bova):
                bova11_patrimonio.append(np.nan)
            else:
                qtd_bova = valor_aporte // preco_bova
                bova11_quantidade += qtd_bova
                patrimonio_bova = bova11_quantidade * preco_bova
                bova11_patrimonio.append(patrimonio_bova)
            historico_pesos.append({})
            historico_num_ativos.append(0)
            continue

        returns = lookback_prices.pct_change().dropna()

        if opt_method == "Sharpe (macro)":
            pesos = otimizar_carteira_sharpe(
                ativos_validos_tickers, carteira, favorecimentos=favorecimentos
            )
        elif opt_method == "HRP":
            pesos = otimizar_carteira_hrp(
                ativos_validos_tickers, carteira, favorecimentos=favorecimentos
            )
        else:
            raise ValueError("Método de otimização desconhecido.")

        pesos = pd.Series(pesos).clip(upper=limite_porc_ativo)
        if pesos.sum() > 0:
            pesos = pesos / pesos.sum()
        else:
            pesos = pd.Series(1.0 / len(ativos_validos_tickers), index=ativos_validos_tickers)

        if data_aporte in period_prices.index:
            precos_mes = period_prices.loc[data_aporte, ativos_validos_tickers]
        else:
            precos_mes = period_prices.loc[period_prices.index.asof(data_aporte), ativos_validos_tickers]
        valores_ativos_atuais = {ativo: carteira.get(ativo, 0) * precos_mes[ativo] for ativo in ativos_validos_tickers}

        total_carteira = sum(valores_ativos_atuais.values())
        total_novo = total_carteira + valor_aporte
        valor_total_alvo = {ativo: pesos[ativo] * total_novo for ativo in ativos_validos_tickers}
        valor_alocar = {ativo: max(0.0, valor_total_alvo[ativo] - valores_ativos_atuais.get(ativo, 0)) for ativo in ativos_validos_tickers}

        for ativo in ativos_validos_tickers:
            valor_compra = valor_alocar[ativo]
            if valor_compra > 0 and precos_mes[ativo] > 0:
                qtd = int(valor_compra // precos_mes[ativo])
                carteira[ativo] = carteira.get(ativo, 0) + qtd

        data_ultima = period_prices.index.asof(data_aporte + pd.offsets.MonthEnd(0))
        precos_fim = period_prices.loc[data_ultima, ativos_validos_tickers]
        patrimonio = sum(carteira.get(ativo, 0) * precos_fim[ativo] for ativo in ativos_validos_tickers)
        valor_carteira.append(patrimonio)
        datas_carteira.append(data_ultima)
        historico_pesos.append(pesos.to_dict())
        historico_num_ativos.append(len(ativos_validos_tickers))

        preco_bova = bova11_prices.asof(data_aporte)
        if np.isnan(preco_bova):
            bova11_patrimonio.append(np.nan)
        else:
            qtd_bova = valor_aporte // preco_bova
            bova11_quantidade += qtd_bova
            patrimonio_bova = bova11_quantidade * preco_bova
            bova11_patrimonio.append(patrimonio_bova)

    df_result = pd.DataFrame({
        'Carteira HRPMACRO': valor_carteira,
        'BOVA11': bova11_patrimonio
    }, index=datas_carteira)

    st.line_chart(df_result)
    st.write("Evolução da carteira (HRPMACRO) vs Benchmark (BOVA11):")
    st.write(df_result)

    # Exibir tabela dos indicadores macroeconômicos utilizados em cada mês
    st.subheader("Histórico dos Indicadores Macroeconômicos Utilizados")
    df_macro = pd.DataFrame(historico_macro)
    df_macro.set_index('data', inplace=True)
    st.dataframe(df_macro, use_container_width=True)

    n_years = (df_result.index[-1] - df_result.index[0]).days / 365.25
    total_aportado = valor_aporte * len(datas_aporte)
    carteira_cagr = (df_result['Carteira HRPMACRO'].iloc[-1] / total_aportado) ** (1/n_years) - 1
    bova_cagr = (df_result['BOVA11'].iloc[-1] / total_aportado) ** (1/n_years) - 1
    st.metric("CAGR Carteira HRPMACRO", f"{carteira_cagr:.2%}")
    st.metric("CAGR BOVA11", f"{bova_cagr:.2%}")
    st.write("Número de ativos por mês:", historico_num_ativos)
    st.write("Pesos por mês:", historico_pesos)

    # Mostrar composição final da carteira ao término da simulação
    st.subheader("Carteira Final no Término da Simulação")
    final_prices = precos.loc[df_result.index[-1], [t for t in carteira.keys() if t in precos.columns]]
    carteira_final = []
    for ativo, qtd in carteira.items():
        if ativo in final_prices and qtd > 0:
            valor = qtd * final_prices[ativo]
            carteira_final.append({
                "Ativo": ativo,
                "Quantidade": int(qtd),
                "Preço Final": final_prices[ativo],
                "Valor Final (R$)": valor
            })
    df_carteira_final = pd.DataFrame(carteira_final)
    if not df_carteira_final.empty:
        df_carteira_final["% da Carteira"] = 100 * df_carteira_final["Valor Final (R$)"] / df_carteira_final["Valor Final (R$)"].sum()
        st.dataframe(df_carteira_final.sort_values("Valor Final (R$)", ascending=False), use_container_width=True)
    else:
        st.write("Nenhum ativo na carteira final.")

    st.success("Backtest mensal com aportes, rebalanceamento, macro histórico e carteira final exibidos!")

st.caption("Backtest mensal usando cenário macroeconômico histórico real de cada mês para as decisões e otimização. Limite de 30% por ativo, long only, benchmark: BOVA11.")

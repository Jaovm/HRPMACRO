import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from HRPMACRO import (
    setores_por_ticker,
    setores_por_cenario,
    obter_preco_diario_ajustado,
    obter_macro,
    pontuar_macro,
    classificar_cenario_macro,
    calcular_favorecimento_continuo,
    otimizar_carteira_sharpe,
    otimizar_carteira_hrp,
)

st.title("Backtest Mensal com Aportes – HRPMACRO (Long Only, max 30% por ativo)")

# Configuração inicial
valor_aporte = 1000.0
limite_porc_ativo = 0.3  # 30%
start_date = pd.to_datetime("2018-01-01")
end_date = pd.to_datetime(datetime.date.today())

# Seleção manual dos ativos
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

    # Download histórico de preços (ajustado)
    st.write("Baixando preços históricos dos ativos e benchmark (BOVA11.SA)...")
    all_tickers = list(set(tickers + ['BOVA11.SA']))
    precos = obter_preco_diario_ajustado(all_tickers)
    precos = precos.ffill().dropna()

    carteira = {t: 0 for t in tickers}
    caixa = 0.0

    # Para benchmark DCA no BOVA11
    bova11_prices = precos['BOVA11.SA']
    bova11_quantidade = 0
    bova11_patrimonio = []

    for idx, data_aporte in enumerate(datas_aporte):
        st.write(f"Processando mês: {data_aporte.strftime('%Y-%m')}")

        data_fim_mes = data_aporte + pd.DateOffset(months=1) - pd.Timedelta(days=1)
        data_fim_mes = min(data_fim_mes, end_date)
        period_prices = precos.loc[:data_fim_mes].copy()

        # Macro e favorecimento
        macro = obter_macro()
        score_macro = pontuar_macro(macro)
        cenario = classificar_cenario_macro(
            ipca=macro.get("ipca"),
            selic=macro.get("selic"),
            dolar=macro.get("dolar"),
            pib=macro.get("pib"),
            preco_soja=macro.get("soja"),
            preco_milho=macro.get("milho"),
            preco_minerio=macro.get("minerio"),
            preco_petroleo=macro.get("petroleo"),
        )
        ativos_validos = []
        for t in tickers:
            setor = setores_por_ticker.get(t)
            if setor is None or t not in period_prices.columns:
                continue
            favorecido = calcular_favorecimento_continuo(setor, score_macro)
            ativos_validos.append(
                {"ticker": t, "setor": setor, "favorecido": favorecido}
            )
        if not ativos_validos:
            st.warning(f"Nenhum ativo válido em {data_aporte.strftime('%Y-%m')}. Pulando mês.")
            valor_carteira.append(patrimonio)
            datas_carteira.append(data_aporte)
            # Benchmark DCA
            preco_bova = bova11_prices.asof(data_aporte)
            if np.isnan(preco_bova):
                bova11_patrimonio.append(np.nan)
            else:
                qtd_bova = valor_aporte // preco_bova
                bova11_quantidade += qtd_bova
                patrimonio_bova = bova11_quantidade * preco_bova
                bova11_patrimonio.append(patrimonio_bova)
            continue

        ativos_validos_tickers = [a["ticker"] for a in ativos_validos]
        favorecimentos = {a["ticker"]: a["favorecido"] for a in ativos_validos}

        lookback_inicio = data_aporte - pd.DateOffset(months=12)
        lookback_prices = precos.loc[lookback_inicio:data_aporte, ativos_validos_tickers].dropna()
        if len(lookback_prices) < 2:
            st.warning(f"Dados insuficientes para otimização em {data_aporte.strftime('%Y-%m')}. Pulando mês.")
            valor_carteira.append(patrimonio)
            datas_carteira.append(data_aporte)
            # Benchmark DCA
            preco_bova = bova11_prices.asof(data_aporte)
            if np.isnan(preco_bova):
                bova11_patrimonio.append(np.nan)
            else:
                qtd_bova = valor_aporte // preco_bova
                bova11_quantidade += qtd_bova
                patrimonio_bova = bova11_quantidade * preco_bova
                bova11_patrimonio.append(patrimonio_bova)
            continue

        returns = lookback_prices.pct_change().dropna()

        # Otimização
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

        # Limite de 30% por ativo e normalização
        pesos = pd.Series(pesos).clip(upper=limite_porc_ativo)
        if pesos.sum() > 0:
            pesos = pesos / pesos.sum()
        else:
            pesos = pd.Series(1.0 / len(ativos_validos_tickers), index=ativos_validos_tickers)

        # Preços do mês para compra
        if data_aporte in period_prices.index:
            precos_mes = period_prices.loc[data_aporte, ativos_validos_tickers]
        else:
            precos_mes = period_prices.loc[period_prices.index.asof(data_aporte), ativos_validos_tickers]
        valores_ativos_atuais = {ativo: carteira.get(ativo, 0) * precos_mes[ativo] for ativo in ativos_validos_tickers}

        # Sugere alocação do novo aporte (proporcional aos pesos alvo)
        total_carteira = sum(valores_ativos_atuais.values())
        total_novo = total_carteira + valor_aporte
        valor_total_alvo = {ativo: pesos[ativo] * total_novo for ativo in ativos_validos_tickers}
        valor_alocar = {ativo: max(0.0, valor_total_alvo[ativo] - valores_ativos_atuais.get(ativo, 0)) for ativo in ativos_validos_tickers}

        # Atualiza quantidades da carteira
        for ativo in ativos_validos_tickers:
            valor_compra = valor_alocar[ativo]
            if valor_compra > 0 and precos_mes[ativo] > 0:
                qtd = int(valor_compra // precos_mes[ativo])
                carteira[ativo] = carteira.get(ativo, 0) + qtd

        # Atualiza patrimônio com preços do fim do mês
        data_ultima = period_prices.index.asof(data_fim_mes)
        precos_fim = period_prices.loc[data_ultima, ativos_validos_tickers]
        patrimonio = sum(carteira.get(ativo, 0) * precos_fim[ativo] for ativo in ativos_validos_tickers)
        valor_carteira.append(patrimonio)
        datas_carteira.append(data_ultima)
        historico_pesos.append(pesos.to_dict())
        historico_num_ativos.append(len(ativos_validos_tickers))

        # Benchmark DCA
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

    # Métricas finais
    n_years = (df_result.index[-1] - df_result.index[0]).days / 365.25
    total_aportado = valor_aporte * len(datas_aporte)
    carteira_cagr = (df_result['Carteira HRPMACRO'].iloc[-1] / total_aportado) ** (1/n_years) - 1
    bova_cagr = (df_result['BOVA11'].iloc[-1] / total_aportado) ** (1/n_years) - 1
    st.metric("CAGR Carteira HRPMACRO", f"{carteira_cagr:.2%}")
    st.metric("CAGR BOVA11", f"{bova_cagr:.2%}")
    st.write("Número de ativos por mês:", historico_num_ativos)
    st.write("Pesos por mês:", historico_pesos)
    st.success("Backtest mensal com aportes e rebalanceamento concluído!")

st.caption("Backtest mensal com aportes e rebalanceamento usando otimização macro ou HRP do arquivo HRPMACRO.py. Limite de 30% por ativo, long only, alocação recalculada a cada mês. Benchmark: BOVA11 DCA.")

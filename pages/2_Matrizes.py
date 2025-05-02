import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
from scipy.stats import skew
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("\U0001F4C8 Análise Multicritério de Ações com Fuzzy-TOPSIS")

# Tickers padrão
user_tickers = st.multiselect("Selecione as empresas:", ["TTWO", "TCEHY", "EA", "RBLX", "NCBDF"], default=["TTWO", "EA"])
tickers = user_tickers

@st.cache_data

def collect_data(tickers, start="2019-01-01"):
    tq = Ticker(tickers)
    df = tq.history(start=start).reset_index()
    prices = df.pivot(index="date", columns="symbol", values="adjclose").dropna()
    volume = df.pivot(index="date", columns="symbol", values="volume").dropna()
    return prices, volume

def max_drawdown(series):
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def calculate_beta(asset_returns, market_returns):
    cov = np.cov(asset_returns, market_returns)[0][1]
    var = np.var(market_returns)
    return cov / var if var != 0 else np.nan

def build_matrix(lr, vol):
    mean_ret = lr.mean()
    std_ret = lr.std()
    liq = vol.mean()
    market = lr.mean(axis=1)

    sharpe = mean_ret / std_ret
    drawdowns = lr.apply(max_drawdown)
    annual_ret = (1 + mean_ret) ** 252 - 1
    beta = lr.apply(lambda x: calculate_beta(x, market))
    skewness = lr.apply(skew)
    vol_annual = std_ret * np.sqrt(252)

    return pd.DataFrame({
        'retorno': mean_ret,
        'retorno_anual': annual_ret,
        'sharpe': sharpe,
        'vol_anual': vol_annual,
        'drawdown': drawdowns,
        'beta': beta,
        'skewness': skewness,
        'liquidez': liq,
        'correlacao': lr.corrwith(market)
    })

def normalize_df(df):
    nd = df.copy()
    for c in df.columns:
        if c in ['vol_anual', 'drawdown', 'beta', 'correlacao']:
            nd[c] = df[c].min() / df[c]
        elif c == 'skewness':
            nd[c] = df[c].abs().min() / df[c].abs()
        else:
            nd[c] = df[c] / df[c].max()
    return nd

def fuzzy_topsis(R, weights):
    results = {}
    for cond, w in weights.items():
        cc = pd.DataFrame(index=R.index, columns=['inferior', 'modal', 'superior'])
        for lvl, w_factor in zip(cc.columns, [0.9, 1.0, 1.1]):
            W = w * w_factor
            M = R.values * W
            pos, neg = M.max(0), M.min(0)
            d_pos = np.linalg.norm(M - pos, axis=1)
            d_neg = np.linalg.norm(M - neg, axis=1)
            cc[lvl] = d_neg / (d_pos + d_neg)
        results[cond] = cc
    return results

def plot_heatmap(df, title):
    fig = px.imshow(df.T, text_auto=".2f", aspect="auto",
                    color_continuous_scale='Viridis',
                    labels=dict(x="Ativo", y="Critério", color="Valor"))
    fig.update_layout(title=title, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def plot_radar(df, ativos):
    fig = go.Figure()
    for ativo in ativos:
        fig.add_trace(go.Scatterpolar(
            r=df.loc[ativo].values,
            theta=df.columns,
            fill='toself',
            name=ativo
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                      title="Radar dos Ativos", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# Execução
prices, volume = collect_data(tickers)
log_returns = np.log(prices / prices.shift(1)).dropna()
decision_df = build_matrix(log_returns, volume)
R = normalize_df(decision_df)

weights = {
    'baixa':        np.array([0.10, 0.05, 0.12, 0.04, 0.10, 0.10, 0.05, 0.30, 0.14]),
    'estabilidade': np.array([0.05, 0.10, 0.08, 0.14, 0.08, 0.15, 0.10, 0.10, 0.20]),
    'alta':         np.array([0.20, 0.15, 0.20, 0.15, 0.10, 0.05, 0.05, 0.05, 0.05])
}

results = fuzzy_topsis(R, weights)
probs = {'baixa': 0.3, 'estabilidade': 0.5, 'alta': 0.2}
exp_cc = pd.Series(0, index=results['baixa'].index)
for cond, p in probs.items():
    exp_cc += results[cond]['modal'] * p
ranking = exp_cc.sort_values(ascending=False)

# Layout
st.subheader("Matriz de Decisão (Original)")
st.dataframe(decision_df.style.format("{:.4f}"))

st.subheader("Matriz Normalizada")
st.dataframe(R.style.format("{:.4f}"))

st.subheader("Visualização da Matriz")
plot_heatmap(R, "Matriz de Decisão Normalizada")

st.subheader("Radar Chart dos Top 5")
plot_radar(R, ativos=ranking.index[:5])

st.subheader("Ranking Final (Fuzzy-TOPSIS)")
st.dataframe(ranking.reset_index().rename(columns={0: "CC Esperado"}))

st.subheader("Análise Fuzzy por Cenário")
for cond in results:
    st.write(f"### Cenário: {cond.capitalize()}")
    st.dataframe(results[cond].style.format("{:.4f}"))

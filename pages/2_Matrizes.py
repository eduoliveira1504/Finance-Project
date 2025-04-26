# pages/2_Analise_TOPSIS.py

import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Análise Fuzzy TOPSIS", layout="wide")

st.title("Análise Fuzzy TOPSIS dos Ativos")

# Lista de tickers
tickers = ["TTWO", "TCEHY", "EA", "RBLX", "NCBDF"]

@st.cache_data
def collect_data(tickers, start="2019-01-01"):
    tq = Ticker(tickers)
    df = tq.history(start=start).reset_index()
    prices = df.pivot(index="date", columns="symbol", values="adjclose").dropna()
    return prices

prices = collect_data(tickers)
log_returns = np.log(prices / prices.shift(1)).dropna()

@st.cache_data
def get_volume(tickers, start="2019-01-01"):
    tq = Ticker(tickers)
    df = tq.history(start=start).reset_index()
    volume = df.pivot(index="date", columns="symbol", values="volume").dropna()
    return volume

volume = get_volume(tickers)

def build_matrix(lr, vol):
    mean_ret = lr.mean()
    std_ret = lr.std()
    liq = vol.mean()
    market = lr.mean(axis=1)
    corr = lr.corrwith(market)
    return pd.DataFrame({
        'retorno': mean_ret,
        'risco': std_ret,
        'liquidez': liq,
        'correlacao': corr
    })

decision_df = build_matrix(log_returns, volume)

def normalize_df(df):
    nd = df.copy()
    for c in df.columns:
        if c in ['risco','correlacao']:
            nd[c] = df[c].min() / df[c]
        else:
            nd[c] = df[c] / df[c].max()
    return nd

R = normalize_df(decision_df)

weights = {
    'baixa':        np.array([0.10,0.35,0.45]),
    'estabilidade': np.array([0.02,0.10,0.12]),
    'alta':         np.array([0.04,0.10,0.22])
}

def fuzzy_topsis(R, weights):
    results = {}
    for cond, w_ret in weights.items():
        w_risk = 1 - w_ret
        cc = pd.DataFrame(index=R.index, columns=['inferior','modal','superior'])
        for lvl,(wr,wk) in zip(cc.columns, zip(w_ret, w_risk)):
            M = R[['retorno','risco']].values * np.array([wr,wk])
            pos, neg = M.max(0), M.min(0)
            d_pos = np.linalg.norm(M-pos, axis=1)
            d_neg = np.linalg.norm(M-neg,axis=1)
            cc[lvl] = d_neg/(d_pos+d_neg)
        results[cond] = cc
    return results

results = fuzzy_topsis(R, weights)

probs = {'baixa':0.3,'estabilidade':0.5,'alta':0.2}
exp_cc = pd.Series(0, index=results['baixa'].index)
for cond,p in probs.items():
    exp_cc += results[cond]['modal'] * p
exp_cc = exp_cc.sort_values(ascending=False)

# Mostrar Tabela de Resultados
st.subheader("Ranking Esperado dos Ativos")
st.dataframe(exp_cc)

st.subheader("Resultados Fuzzy TOPSIS por Condição")

def interpret_cc(results):
    for cond, df in results.items():
        d = df.reset_index().rename(columns={'symbol':'ativo'})
        fig = px.bar(
            d,
            x='ativo',
            y='modal',
            error_y=d['superior'] - d['modal'],
            error_y_minus=d['modal'] - d['inferior'],
            title=f"Condição {cond.capitalize()} — CC_modal com barras fuzzy",
            labels={"modal":"CC_modal","ativo":"Ativo"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

interpret_cc(results)


import streamlit as st
import yahooquery as yq
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(layout="wide")
st.title("🤖 Previsão de Preços com Machine Learning")

symbols = ["TTWO", "TCEHY", "EA", "RBLX", "NCBDF"]

selected_stock = st.selectbox("Selecione a empresa para previsão", symbols)

periods = {"3 meses": "3mo", "6 meses": "6mo", "1 ano": "1y", "2 anos": "2y"}
period_label = st.selectbox("Selecione o período histórico", list(periods.keys()), index=2)
selected_period = periods[period_label]

@st.cache_data
def get_stock_history(symbol, period):
    ticker = yq.Ticker(symbol)
    df = ticker.history(period=period).reset_index()
    df = df[df['symbol'] == symbol]
    return df

df = get_stock_history(selected_stock, selected_period)

if df.empty or len(df) < 10:
    st.warning("Dados insuficientes para treinar o modelo. Tente outro período ou empresa.")
    st.stop()

# ⬇️ Remoção de timezone das datas
df = df[["date", "close"]].dropna()
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
df["days"] = (df["date"] - df["date"].min()).dt.days

# Treinamento do modelo
X = df[["days"]]
y = df["close"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Previsão futura
st.subheader("📅 Previsão Futura")
dias_futuros = st.slider("Quantos dias você quer prever para frente?", 5, 30, 7)
ultimo_dia = df["days"].max()
dias_para_prever = np.array(range(ultimo_dia + 1, ultimo_dia + dias_futuros + 1)).reshape(-1, 1)
previsoes_futuras = model.predict(dias_para_prever)

# Datas futuras sem timezone
datas_futuras = [df["date"].max() + pd.Timedelta(days=i) for i in range(1, dias_futuros + 1)]
datas_futuras = pd.to_datetime(datas_futuras).tz_localize(None)

# Gráfico
st.subheader("📉 Preço Real + Previsão Futura")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["date"],
    y=df["close"],
    mode='lines+markers',
    name='Preço Real',
    line=dict(color='blue', width=2)
))

fig.add_trace(go.Scatter(
    x=datas_futuras,
    y=previsoes_futuras,
    mode='lines+markers',
    name='Previsão Futura',
    line=dict(color='orange', width=2)
))

fig.update_layout(
    title=f"Previsão de preços - {selected_stock}",
    xaxis_title="Data",
    yaxis_title="Preço",
    legend_title="Legenda",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# Tabela com tipos corretos
previsao_df = pd.DataFrame({
    "Data": pd.to_datetime(datas_futuras),
    "Preço Previsto": previsoes_futuras.astype(float)
})

st.metric("Erro médio (RMSE)", f"${rmse:.2f}")
st.dataframe(previsao_df, use_container_width=True)

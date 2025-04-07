import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("📈 Previsão de valores com Machine Learning")

# Input do usuário
ticker = st.text_input("Digite o código do ativo (ex: PETR4.SA)", "PETR4.SA")
periodo = st.selectbox("Selecione o período de histórico", ["1y", "2y", "5y"])
dias_futuros = st.slider("Dias para prever no futuro", 7, 90, 30)

# Buscar dados
ticker_obj = Ticker(ticker)
data = ticker_obj.history(period=periodo)

if data.empty:
    st.error("Não foi possível carregar os dados para esse ticker.")
    st.stop()

# Resetar índice e garantir formato de data correto
data = data.reset_index()
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Remover fuso horário, se houver
if data['date'].dt.tz is not None:
    data['date'] = data['date'].dt.tz_localize(None)

# Selecionar apenas colunas necessárias
df = data[["date", "close"]].dropna()

# Preparar os dados
df["days"] = (df["date"] - df["date"].min()).dt.days
X = df[["days"]]
y = df["close"]

# Treinamento do modelo
model = LinearRegression()
model.fit(X, y)

# Previsão futura
last_day = df["days"].max()
future_days = np.arange(last_day + 1, last_day + dias_futuros + 1).reshape(-1, 1)
future_dates = [df["date"].max() + pd.Timedelta(days=i) for i in range(1, dias_futuros + 1)]
future_preds = model.predict(future_days)

# Gráfico
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["date"], df["close"], label="Histórico", color="blue")
ax.plot(future_dates, future_preds, label="Previsão", color="green")
ax.set_title(f"Previsão para {ticker}")
ax.set_xlabel("Data")
ax.set_ylabel("Valor de fechamento")
ax.legend()
st.pyplot(fig)

# Tabela com previsões futuras
df_future = pd.DataFrame({
    "Data": future_dates,
    "Valor previsto": future_preds
})
df_future["Data"] = df_future["Data"].dt.strftime("%Y-%m-%d")
st.subheader("📅 Previsão futura")
st.dataframe(df_future)

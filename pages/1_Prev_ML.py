import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import timedelta

st.set_page_config(page_title="Previsão de Ações", layout="wide")
st.title("📈 Previsão de Ações com Regressão Linear")

# Inputs do usuário
symbol = st.text_input("Digite o código da ação (ex: PETR4.SA):", "PETR4.SA")
period = st.selectbox("Selecione o período de dados históricos:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
predict_days = st.slider("Dias futuros para prever:", 1, 30, 7)

# Coleta de dados
ticker = Ticker(symbol)
data = ticker.history(period=period)

if data.empty:
    st.warning("Não foi possível obter dados para esse papel. Verifique o código e tente novamente.")
    st.stop()

# Resetar index caso necessário
if isinstance(data.index, pd.MultiIndex):
    data = data.reset_index()

# Conversão de fuso horário (evitar erro de tz-aware vs tz-naive)
if data['date'].dt.tz is not None:
    data['date'] = data['date'].dt.tz_localize(None)

# Selecionar apenas colunas necessárias
df = data[["date", "close"]].dropna()
df["date"] = pd.to_datetime(df["date"])
df["days"] = (df["date"] - df["date"].min()).dt.days

# Features e target
X = df[["days"]]
y = df["close"]

# Verificar quantidade de dados
if len(X) < 10:
    st.warning("Dados insuficientes para treinar o modelo. Tente outro período ou papel.")
    st.stop()

# Separar dados e treinar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)

# Previsão futura
last_day = df["days"].max()
future_days = np.array(range(last_day + 1, last_day + predict_days + 1)).reshape(-1, 1)
future_dates = [df["date"].max() + timedelta(days=i) for i in range(1, predict_days + 1)]
future_preds = model.predict(future_days)

# Plotando o gráfico
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["date"], df["close"], label="Histórico")
ax.plot(future_dates, future_preds, label="Previsão", color="orange")
ax.set_xlabel("Data")
ax.set_ylabel("Preço de Fechamento")
ax.set_title(f"Preço da ação {symbol}")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Exibir gráfico
st.pyplot(fig)

# Exibir dados futuros
future_df = pd.DataFrame({"Data": future_dates, "Preço Previsto": future_preds})
st.subheader("📊 Tabela de Previsão Futura")
st.dataframe(future_df, hide_index=True, use_container_width=True)

# Observação
st.caption("Esta previsão é baseada em um modelo de regressão linear simples e não deve ser usada para decisões financeiras.")

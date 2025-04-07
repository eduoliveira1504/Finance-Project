import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from yahooquery import Ticker

st.set_page_config(page_title="Previsão de Bolsa", layout="wide")

st.title("📈 Previsão de Valores da Bolsa com Machine Learning")

# ⬇️ Entrada do usuário
ticker_symbol = st.text_input("Digite o código da ação (ex: PETR4.SA):", "PETR4.SA")

# ⬇️ Coleta de dados com yahooquery
ticker = Ticker(ticker_symbol)
try:
    df = ticker.history(period="1y").reset_index()
except Exception as e:
    st.error(f"Erro ao buscar dados: {e}")
    st.stop()

# ⬇️ Limpeza e preparação dos dados
df = df[["date", "close"]].dropna()
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 🔧 Correção de timezone de forma segura
df["date"] = df["date"].apply(lambda x: x.tz_convert(None) if hasattr(x, "tzinfo") and x.tzinfo else x)

df = df.dropna(subset=["date"])  # Remove linhas que falharam a conversão
df["days"] = (df["date"] - df["date"].min()).dt.days

# ⬇️ Treinamento do modelo
X = df[["days"]]
y = df["close"]
model = LinearRegression()
model.fit(X, y)

# ⬇️ Previsão para os próximos 30 dias
future_days = 30
last_day = df["days"].max()
future_X = pd.DataFrame({"days": np.arange(last_day + 1, last_day + future_days + 1)})
future_preds = model.predict(future_X)

# ⬇️ Criação do DataFrame futuro
future_dates = df["date"].max() + pd.to_timedelta(future_X["days"] - last_day, unit="D")
future_df = pd.DataFrame({
    "date": future_dates,
    "close": future_preds
})

# ⬇️ Combina dados reais com previsão
combined_df = pd.concat([df[["date", "close"]], future_df], ignore_index=True)

# ⬇️ Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["date"], df["close"], label="Histórico")
ax.plot(future_df["date"], future_df["close"], linestyle="--", label="Previsão", color="orange")
ax.set_title(f"Previsão de Fechamento para {ticker_symbol}")
ax.set_xlabel("Data")
ax.set_ylabel("Valor (R$)")
ax.legend()
st.pyplot(fig)

# ⬇️ Exibir dados
with st.expander("📊 Ver dados"):
    st.dataframe(combined_df)

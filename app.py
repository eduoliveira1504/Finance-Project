import streamlit as st
import yahooquery as yq
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuração inicial da página
st.set_page_config(layout="wide", page_title="Dashboard Financeiro Interativo")
st.title("📊 Dashboard Financeiro Interativo")

# Lista de símbolos das empresas
symbols = ["TTWO", "TCEHY", "EA", "RBLX", "NCBDF"]

# Funções para obter dados de ações
@st.cache_data(ttl=3600)
def get_stock_data(symbol, period):
    stock = yq.Ticker(symbol)
    hist = stock.history(period=period).reset_index()
    return hist

@st.cache_data(ttl=3600)
def get_financials(symbol):
    stock = yq.Ticker(symbol)
    return stock.summary_detail.get(symbol, {})

@st.cache_data(ttl=300)
def get_current_price(symbol):
    stock = yq.Ticker(symbol)
    return stock.price.get(symbol, {}).get("regularMarketPrice", "N/A")

@st.cache_data(ttl=300)
def get_previous_close(symbol):
    stock = yq.Ticker(symbol)
    return stock.price.get(symbol, {}).get("regularMarketPreviousClose", "N/A")

# Barra lateral de configurações
st.sidebar.header("🔍 Configurações")

time_periods = {
    "1 mês": "1mo",
    "3 meses": "3mo",
    "6 meses": "6mo",
    "1 ano": "1y",
    "5 anos": "5y"
}
selected_period = st.sidebar.selectbox("Selecione o período histórico", list(time_periods.keys()), index=3)
graph_type = st.sidebar.selectbox("Selecione o tipo de gráfico", ["Linha", "Candlestick"])

# Seleção de ações
if graph_type == "Candlestick":
    selected_stock = st.sidebar.selectbox("Selecione uma empresa", symbols)
    selected_stocks = [selected_stock]
else:
    selected_stocks = st.sidebar.multiselect("Selecione as empresas para comparação", symbols, default=symbols)

# Preço atual e fechamento anterior com cores na barra lateral
price_data = {stock: get_current_price(stock) for stock in symbols}
previous_close_data = {stock: get_previous_close(stock) for stock in symbols}

for stock in symbols:
    current = price_data.get(stock, "N/A")
    previous = previous_close_data.get(stock, "N/A")

    # Lógica de cor condicional
    if isinstance(current, (int, float)) and isinstance(previous, (int, float)):
        color = "#28a745" if current > previous else "#dc3545" if current < previous else "#ffc107"
    else:
        color = "#6c757d"

    st.sidebar.markdown(
        f"<span style='color: {color}; font-weight: bold;'>{stock}: ${current}</span>",
        unsafe_allow_html=True
    )

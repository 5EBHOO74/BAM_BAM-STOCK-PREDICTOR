import streamlit as st
from datetime import date
import yfinance as yf
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from sklearn import preprocessing
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric


START = "2012-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('BAM PREDICTOR APP')

stocks = ("SNAP","SHOP","AMD","TEVA","AAPL","F","CCL","ITUB","NVDA","AMZN","GOOGL","HTA","META","PGY","MSFT","BBD","T","ABEV","GOOG","AAL","NIO","INTC","PBR","PYPL","BAC","VG","KGC","SIRI","CS","PSTH","XOM","COIN","AFRM","CSCO","FCX","SOFI","V","AUY","FTCH","WBD","PFE","BA","CSX","OXY","LCID","C","GGB","CVE","VTRS","BTG","WMT","SQ","KMI","DKNG","BEKE","QCOM","RIG","UMC","PLUG","GRAB","GM","AGNC","MRO","CCJ","KO","NYCB","CPG","MU","TXN","TWTR","ENPH","MRVL","SLB","LYFT","SNDL","TELL","UAL","BKR","NFLX","ONEM","RIVN","DAL","CVNA","BP","JNPR","BMY","IBN","AMCR","DIS","TSM","MO","JPM","DIDIY","BTC","LYG","RUN","PTON","ETH","MSFT","GME")
chosen_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider("Yearly prediction interval:", 1,5)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(chosen_stock)
data_load_state.text('Processing data.done!')

st.subheader('Unprocessed data')
st.write(data.tail())

# Normalizing Data
data['Close_norm'] = preprocessing.normalize([data['Close'].values])[0]
data['Open_norm'] = preprocessing.normalize([data['Open'].values])[0]


# Data Understanding
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data ', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()


# Train the Data
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Model fitting
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast Results')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Prediction components")
fig2 = m.plot_components(forecast)
st.write(fig2)

st.write("MEASURING PERFORMANCE")
cv = cross_validation(model=m, initial='500 days', period = '30 days', horizon = '100 days')
st.write(cv.head(5))

st.write("PERFORMANCE METRICS")
p = performance_metrics(cv)
st.write(p.head(5))

st.write("Cross-Validation with MAPE")
fig3 = plot_cross_validation_metric(cv, metric='mape')
st.write(fig3)
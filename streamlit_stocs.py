import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import streamlit as st
from stocknews import StockNews

st.title('Stock Dashboard')
ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

if ticker != '':
    data = yf.download(ticker, start_date, end_date)
    fig = px.line(data, x=data.index, y=data['Adj Close'], title=ticker)
    st.plotly_chart(fig)

    pricing_data, news = st.tabs(
        ["Pricing Data", "News"])

    with pricing_data:
        data['% Change'] = data["Adj Close"] / data["Adj Close"].shift(1) - 1
        annual_return = data["% Change"].mean()*252*100
        standard_deviation = np.std(data["% Change"])*np.sqrt(252)*100
        risk_adj = annual_return / (standard_deviation * 100)

        st.write("Annual return is: ", annual_return, " %")
        st.write("Standard deviation is: ", standard_deviation, " %")
        st.write("Risk Adj. Return is: ", risk_adj)

        data.dropna(inplace=True)
        st.write(data)

    with news:
        st.header(f'News for {ticker}')
        stock_news = StockNews(ticker, save_news=False)
        df_news = stock_news.read_rss()

        for i in range(10):
            st.subheader(f"{i + 1}. {df_news["title"][i]}")
            st.write(df_news["published"][i][:-9])
            st.write(df_news["summary"][i])
            st.write("Title sentiment: ", df_news["sentiment_title"]
                     [i], "  News sentiment: ", df_news["sentiment_summary"][i])

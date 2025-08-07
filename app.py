import datetime
import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import numpy as np
import spacy
from transformers import pipeline
import requests

RISK_FREE_RATE = 0.01
TRADING_DAYS = 252
SENTIMENT_ADJUSTMENT = 0.02
NEWS_API_KEY = ""

st.set_page_config(page_title="CAPM", page_icon="", layout='wide')
st.title("Capital Asset Pricing Model (CAPM)")

spacy_nlp = spacy.load("en_core_web_sm")

stocks_available = ['TSLA', 'AAPL', 'NFLX', 'MSFT', 'MGM', 'AMZN', 'NVDA', 'GOOGL']
selected_stocks = st.multiselect("Choose exactly 4 Stocks", stocks_available, max_selections=4)
years = st.number_input("Number of Years", 1, 10, value=3)

if len(selected_stocks) != 4:
    st.warning("Please select exactly 4 stocks.")
    st.stop()

end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=years * 365)

try:
    sp500 = web.DataReader('sp500', 'fred', start_date, end_date)
    sp500 = sp500.dropna()
except Exception as e:
    st.error(f"Error downloading S&P 500 data: {e}")
    st.stop()

try:
    stocks_data = yf.download(selected_stocks, start=start_date, end=end_date)['Close']
    stocks_data = stocks_data.dropna()
except Exception as e:
    st.error(f"Error downloading stocks data: {e}")
    st.stop()

stocks_returns = stocks_data.pct_change().dropna()
sp500_returns = sp500['sp500'].pct_change().dropna()
combined = stocks_returns.join(sp500_returns, how='inner').dropna()

def calculate_capm(stock_ret, market_ret, rf=RISK_FREE_RATE):
    cov_matrix = np.cov(stock_ret, market_ret)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    rm = market_ret.mean() * TRADING_DAYS
    expected_return = rf + beta * (rm - rf)
    return beta, expected_return

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_sentiment_model()

def fetch_news_headlines(stock, api_key=NEWS_API_KEY):
    if not api_key:
        st.warning("No NewsAPI key set. Skipping news fetch.")
        return []
    url = f"https://newsapi.org/v2/everything?q={stock}&apiKey={api_key}&pageSize=5&sortBy=publishedAt"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return [article["title"] for article in articles if "title" in article]
    except Exception as e:
        st.warning(f"Error fetching news: {e}")
        return []

def adjust_return_by_sentiment(stock, expected_return):
    headlines = fetch_news_headlines(stock)
    if not headlines:
        return expected_return
    sentiments = sentiment_analyzer(headlines)
    scores = {"POSITIVE": 0, "NEGATIVE": 0}
    for result in sentiments:
        scores[result['label']] += result['score']
    net_score = scores["POSITIVE"] - scores["NEGATIVE"]
    adjustment = SENTIMENT_ADJUSTMENT * net_score
    adjusted_return = np.clip(expected_return + adjustment, 0, 1)
    return adjusted_return

apply_sentiment = st.checkbox("Apply AI-Based Sentiment Adjustment to Expected Returns")

results = []
for stock in selected_stocks:
    beta, exp_return = calculate_capm(combined[stock], combined['sp500'])
    ai_return = adjust_return_by_sentiment(stock, exp_return) if apply_sentiment else exp_return
    results.append({
        "Stock": stock,
        "Beta": beta,
        "Expected Return": exp_return,
        "AI-Adjusted Return": ai_return
    })

results_df = pd.DataFrame(results)

st.subheader("Ask a Smart Question About Your Stocks")
q = st.text_input("Ask a question (e.g. 'What is the safest stock to invest in?')")

def keyword_lemmas(query):
    doc = spacy_nlp(query.lower())
    return set(token.lemma_ for token in doc)

if not results_df.empty and q:
    lemmas = keyword_lemmas(q)
    def contains(*words): return any(word in lemmas for word in words)
    if contains("high", "top") and contains("return", "performance"):
        best = results_df.iloc[results_df['Expected Return'].idxmax()]
        st.info(f"{best['Stock']} has the highest expected return of {best['Expected Return']:.2%}.")
    elif contains("low", "least") and contains("return", "performance"):
        worst = results_df.iloc[results_df['Expected Return'].idxmin()]
        st.info(f"{worst['Stock']} has the lowest expected return of {worst['Expected Return']:.2%}.")
    elif contains("low", "least", "safe") and contains("risk", "beta"):
        safest = results_df.iloc[results_df['Beta'].idxmin()]
        st.info(f"{safest['Stock']} has the lowest beta ({safest['Beta']:.2f}), indicating lower risk.")
    elif contains("high", "most") and contains("risk", "beta", "volatile"):
        riskiest = results_df.iloc[results_df['Beta'].idxmax()]
        st.info(f"{riskiest['Stock']} has the highest beta ({riskiest['Beta']:.2f}), indicating high volatility.")
    elif contains("show", "list", "all"):
        st.write(results_df)
    else:
        st.warning("I couldn't understand your question. Try asking about return or beta.")

st.write("CAPM Results", results_df)

st.subheader("Stock Prices Over Time")
st.line_chart(stocks_data)

st.subheader("Daily Returns")
st.line_chart(stocks_returns)

st.subheader("Correlation Matrix")
st.write(stocks_returns.corr())

selected_sentiment_stock = st.selectbox("Choose a stock for sentiment analysis", selected_stocks)
if selected_sentiment_stock:
    st.write(f"Fetching news for {selected_sentiment_stock}...")
    headlines = fetch_news_headlines(selected_sentiment_stock)
    if not headlines:
        st.warning("No news headlines found.")
    else:
        sentiments = sentiment_analyzer(headlines)
        sentiment_scores = {"POSITIVE": 0, "NEGATIVE": 0}
        for i, result in enumerate(sentiments):
            sentiment_scores[result['label']] += result['score']
            st.markdown(f"{headlines[i]} → {result['label']} ({result['score']:.2f})")
        net_score = sentiment_scores["POSITIVE"] - sentiment_scores["NEGATIVE"]
        st.write("Sentiment Summary")
        if net_score > 0.3:
            st.success("Overall sentiment is positive.")
        elif net_score < -0.3:
            st.error("Overall sentiment is negative.")
        else:
            st.info("Overall sentiment is neutral.")

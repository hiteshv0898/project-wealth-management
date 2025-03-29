import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

def fetch_financial_news(ticker):
    try:
        url = f"https://www.google.com/search?q={ticker}+stock+news"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        articles = []
        for item in soup.find_all('a', href=True):
            if 'url' in item['href']:
                articles.append(item.text)

        return articles[:5]
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []

def analyze_sentiment(news_articles):
    sentiment_scores = []
    for article in news_articles:
        analysis = TextBlob(article)
        sentiment_scores.append(analysis.sentiment.polarity)

    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return avg_sentiment
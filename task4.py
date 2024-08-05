import tweepy
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Replace these with your own Twitter API credentials
consumer_key = 'YOUR_CONSUMER_KEY'
consumer_secret = 'YOUR_CONSUMER_SECRET'
access_token = 'YOUR_ACCESS_TOKEN'
access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'

# Authenticate with the Twitter API
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Fetch tweets
def fetch_tweets(query, count=100):
    tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode="extended").items(count)
    data = [[tweet.full_text, tweet.created_at] for tweet in tweets]
    return pd.DataFrame(data, columns=['Text', 'Timestamp'])

# Example: Fetch tweets about a specific topic
df = fetch_tweets("OpenAI", count=200)
print(df.head())

# Function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Apply sentiment analysis to each tweet
df[['Polarity', 'Subjectivity']] = df['Text'].apply(lambda x: pd.Series(analyze_sentiment(x)))
print(df.head())

# Set the style for seaborn
sns.set(style="whitegrid")

# Distribution of polarity
plt.figure(figsize=(10, 6))
sns.histplot(df['Polarity'], bins=30, kde=True)
plt.title('Polarity Distribution')
plt.xlabel('Polarity')
plt.ylabel('Frequency')
plt.show()

# Distribution of subjectivity
plt.figure(figsize=(10, 6))
sns.histplot(df['Subjectivity'], bins=30, kde=True)
plt.title('Subjectivity Distribution')
plt.xlabel('Subjectivity')
plt.ylabel('Frequency')
plt.show()

# Polarity over time
plt.figure(figsize=(10, 6))
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)
df['Polarity'].resample('D').mean().plot()
plt.title('Average Polarity Over Time')
plt.xlabel('Date')
plt.ylabel('Average Polarity')
plt.show()

# Polarity vs. Subjectivity
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Polarity', y='Subjectivity', data=df)
plt.title('Polarity vs. Subjectivity')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()

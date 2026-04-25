import streamlit as st
import pandas as pd
from textblob import TextBlob
from collections import Counter
import re

# Load data
df = pd.read_csv("../data/data.csv")

# Title
st.title("📊 Social Media Engagement Analysis System")

st.write("""
This system analyzes social media engagement data to identify viral content patterns,
understand audience sentiment, and recommend optimal content strategies.
""")

# Show dataset
st.subheader("Dataset Preview")
st.write(df.head())

# =========================
# VIRALITY SCORE
# =========================
df['virality_score'] = (
    2 * df['shares'] +
    2 * df['saves'] +
    df['likes']
)

st.subheader("📊Virality Analysis")

st.bar_chart(df.groupby('topic')['virality_score'].mean())

st.write("Insight: Certain topics consistently achieve higher virality, indicating stronger audience interest.")

# =========================
# SENTIMENT ANALYSIS
# =========================
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['sentiment_score'] = df['comments'].apply(get_sentiment)

df['sentiment_label'] = df['sentiment_score'].apply(
    lambda x: "Relatable" if x > 0 else "Neutral"
)

st.subheader("💬Sentiment Analysis")
st.bar_chart(df['sentiment_label'].value_counts())

st.write("Insight: A large portion of comments are positive, suggesting strong relatability with the audience.")

# =========================
# A/B TESTING
# =========================
st.subheader("🧪A/B Testing Results")

st.write("### Post Length vs Virality")
st.bar_chart(df.groupby('post_length')['virality_score'].mean())

st.write("Insight: One content length performs slightly better, indicating user preference for that format.")

st.write("### Post Time vs Virality")
st.bar_chart(df.groupby('post_time')['virality_score'].mean())

st.write("Insight: Engagement varies across time, helping identify optimal posting windows.")

# =========================
# TREND FORECASTING
# =========================

def extract_keywords(text):
    words = re.findall(r'\b\w+\b', str(text).lower())
    return words

all_words = []

for caption in df['caption']:
    all_words.extend(extract_keywords(caption))

word_counts = Counter(all_words)

stopwords = ['the', 'is', 'and', 'in', 'to', 'of', 'for', 'on', 'a']

filtered_words = {word: count for word, count in word_counts.items() if word not in stopwords}

top_trends = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:5]

trend_df = pd.DataFrame(top_trends, columns=['keyword', 'frequency'])

# =========================
# RECOMMENDATION ENGINE
# =========================
best_topic = df.groupby('topic')['virality_score'].mean().idxmax()
best_time = df.groupby('post_time')['virality_score'].mean().idxmax()
best_length = df.groupby('post_length')['virality_score'].mean().idxmax()

st.subheader("📈 Trend Forecasting")

st.write("Top Emerging Keywords:")
st.write(trend_df)

st.bar_chart(trend_df.set_index('keyword'))

st.write("Insight: Frequently occurring keywords indicate emerging trends that can guide future content strategy.")

st.subheader("📌 Recommended Strategy")

col1, col2, col3 = st.columns(3)

col1.metric("Top Topic", best_topic)
col2.metric("Best Time", best_time)
col3.metric("Best Length", best_length)


st.success(f"""
✅ Final Recommendation:

Focus on **{best_length} {best_topic} content**  
Post during **{best_time} hours**

This strategy maximizes engagement and relatability.
""")
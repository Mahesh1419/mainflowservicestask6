import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Time Series Data
np.random.seed(42)
date_range = pd.date_range(start='1/1/2020', periods=36, freq='M')
sales = np.random.randint(100, 200, size=(36,)) + np.arange(36) * 2
time_series_data = pd.DataFrame({'Date': date_range, 'Sales': sales})

# Time Series Analysis
time_series_data.set_index('Date', inplace=True)
decomposition = seasonal_decompose(time_series_data['Sales'], model='additive')
decomposition.plot()
plt.show()

# ARIMA Model for Forecasting
model = ARIMA(time_series_data['Sales'], order=(1, 1, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=12)
print(forecast)

# Text Data for Sentiment Analysis
sample_texts = [
    "I love this product!",
    "This is the worst service ever.",
    "I'm very happy with my purchase.",
    "I will never buy from this store again."
]

# Sentiment Analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

sentiments = [analyze_sentiment(text) for text in sample_texts]
sentiment_df = pd.DataFrame({'Text': sample_texts, 'Sentiment': sentiments})
print(sentiment_df)

# Data for Clustering
X, _ = make_blobs(n_samples=100, centers=3, cluster_std=1.0, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
clusters = kmeans.labels_

# Plotting the Clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
import tweepy  # For accessing Twitter API
import matplotlib
matplotlib.use('Agg')


# Load Models for Sentiment Analysis
def load_models():
    print("Loading models...")
    models = {
        'distilbert-sentiment': pipeline('sentiment-analysis', model='bhadresh-savani/distilbert-base-uncased-emotion'),
        'bert-sentiment': pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment'),
        'roberta-sentiment': pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment'),
        'cardiffnlp-sentiment': pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment')
    }
    print("Models loaded successfully!")
    return models

# Define Emotion Mapping for Consistency
emotion_mapping = {
    'distilbert-sentiment': {'sadness': 'Negative', 'joy': 'Positive', 'neutral': 'Neutral'},
    'bert-sentiment': {
        '1 star': 'Negative', '2 stars': 'Negative', '3 stars': 'Neutral',
        '4 stars': 'Positive', '5 stars': 'Positive'
    },
    'roberta-sentiment': {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'},
    'cardiffnlp-sentiment': {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}
}

# Analyze Sentiments for Given Text
def analyze_sentiments(models, text, threshold=0.1):  # Lowered threshold to check for low-confidence outputs
    results = {}
    for model_name, model in models.items():
        try:
            analysis = model(text)
            filtered_results = []
            for item in analysis:
                emotion = emotion_mapping.get(model_name, {}).get(item['label'], 'Unknown')
                score = item['score']
                if score >= threshold:
                    filtered_results.append({'emotion': emotion, 'score': score})
            if not filtered_results:  
                filtered_results.append({'emotion': 'No Output', 'score': 0.0})
            results[model_name] = filtered_results
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            results[model_name] = [{'emotion': 'Error', 'score': 0}]
    return results

# Fetch Tweets using Tweepy
def fetch_tweets(keyword, count=10):
    print(f"Fetching tweets for keyword: {keyword}")
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAHohxgEAAAAA0FU1w2yaJ0YriOlEM%2Fm6su75POo%3D7B7kubXwpFizzVokRb5BazGRnANiukZeSI1ks0USvaZkEvrfGl"

    try:
        client = tweepy.Client(bearer_token=bearer_token)
        tweets = client.search_recent_tweets(query=keyword, max_results=count, tweet_fields=["text", "lang"])
        tweet_texts = [tweet.text for tweet in tweets.data if tweet.lang == "en"]
        print(f"Fetched {len(tweet_texts)} tweets.")
        return tweet_texts
    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return []

# Analyze Sentiments of Tweets
def analyze_tweet_sentiments(models, keyword, count=10):
    tweets = fetch_tweets(keyword, count)
    if not tweets:
        print("No tweets fetched or an error occurred.")
        return {}

    tweet_sentiments = {}
    for idx, tweet in enumerate(tweets):
        tweet_sentiments[f"Tweet {idx + 1}"] = analyze_sentiments(models, tweet)

    return tweet_sentiments

# Plot Results in a Grouped Bar Chart
def plot_grouped_bar_chart(results, output_path="static/sentiment_graph.png"):
    emotions = sorted(set(emotion for model_results in results.values() for emotion in [item['emotion'] for item in model_results]))
    model_names = list(results.keys())
    scores_matrix = np.zeros((len(emotions), len(model_names)))

    for model_idx, model_name in enumerate(model_names):
        model_results = results[model_name]
        for item in model_results:
            if item['emotion'] in emotions:
                emotion_idx = emotions.index(item['emotion'])
                scores_matrix[emotion_idx, model_idx] = item['score']

    bar_width = 0.2
    x_indices = np.arange(len(emotions))
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model_name in enumerate(model_names):
        ax.bar(x_indices + i * bar_width, scores_matrix[:, i], width=bar_width, label=model_name)

    ax.set_xlabel('Emotions', fontsize=12)
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Sentiment Analysis Comparison', fontsize=14)
    ax.set_xticks(x_indices + bar_width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(emotions, rotation=45, fontsize=10)
    ax.legend(title='Models', fontsize=10)
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig(output_path)
    plt.close()

# Main Function
if __name__ == "__main__":
    models = load_models()

    print("\nEnter text or keyword to analyze (use 'tweet:<keyword>' for Twitter):")
    user_input = input().strip()

    if not user_input:
        print("No input provided. Exiting...")
    elif user_input.startswith("tweet:"):
        keyword = user_input.replace("tweet:", "").strip()
        tweet_results = analyze_tweet_sentiments(models, keyword, count=10)
        for tweet, sentiment in tweet_results.items():
            print(f"\n{tweet}:")
            for model, data in sentiment.items():
                print(f"  Model: {model}")
                for item in data:
                    print(f"    {item['emotion']}: {item['score']:.4f}")
            plot_grouped_bar_chart(sentiment, title=f"Sentiment Analysis for {tweet}")
    else:
        results = analyze_sentiments(models, user_input)
        print("\nResults:\n")
        for model, data in results.items():
            print(f"Model: {model}")
            for item in data:
                print(f"  {item['emotion']}: {item['score']:.4f}")
        plot_grouped_bar_chart(results)

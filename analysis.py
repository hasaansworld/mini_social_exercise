from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import sqlite3
import pandas as pd

def get_all_content(db_path='database.sqlite'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    all_content = []
    
    # Get content from posts
    cursor.execute("SELECT content FROM Posts")
    for row in cursor.fetchall():
        if row[0]:  # Check if content is not None
            all_content.append(row[0])
    
    # Get content from comments
    cursor.execute("SELECT content FROM Comments")
    for row in cursor.fetchall():
        if row[0]:  # Check if content is not None
            all_content.append(row[0])
    
    conn.close()
    
    return all_content

def top_10_topics(db_path='database.sqlite'):
    # Download necessary NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    # Load data from database
    content_list = get_all_content(db_path)
    print(f"Loaded {len(content_list)} content items from database")
    
    # Get a basic stopword list
    stop_words = set(stopwords.words('english'))
    
    # Add extra words to make our analysis even better
    stop_words.update([
        'would', 'could', 'should', 'might', 'may', 'must', 'can',
        'get', 'let', 'make', 'take', 'give', 'use', 'see', 'know',
        'think', 'want', 'need', 'feel', 'look', 'seem', 'try',
        'keep', 'come', 'go', 'say', 'tell', 'ask', 'find', 'work',
        'call', 'put', 'set', 'become', 'leave', 'turn', 'start',
        'show', 'hear', 'play', 'run', 'move', 'live', 'believe',
        'bring', 'happen', 'write', 'provide', 'sit', 'stand',
        'lose', 'pay', 'meet', 'include', 'continue', 'learn',
        'change', 'lead', 'understand', 'watch', 'follow', 'stop',
        'create', 'speak', 'read', 'allow', 'add', 'spend', 'grow',
        'open', 'walk', 'win', 'offer', 'remember', 'consider',
        'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect',
        'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain',
        'like', 'good', 'great', 'best', 'new', 'old', 'right', 'big',
        'high', 'different', 'small', 'large', 'next', 'early', 'young',
        'important', 'few', 'public', 'bad', 'same', 'able', 'little',
        'sure', 'real', 'true', 'hard', 'long', 'simple', 'easy',
        'strong', 'possible', 'whole', 'free', 'better', 'full', 'special',
        'clear', 'major', 'available', 'likely', 'difficult', 'ready',
        'serious', 'common', 'recent', 'wrong', 'particular', 'certain',
        'personal', 'open', 'red', 'close', 'fine', 'nice', 'perfect',
        'amazing', 'awesome', 'terrible', 'horrible', 'wonderful',
        'time', 'year', 'day', 'week', 'month', 'hour', 'minute',
        'today', 'yesterday', 'tomorrow', 'now', 'later', 'soon',
        'always', 'never', 'sometimes', 'often', 'usually', 'already',
        'still', 'yet', 'ago', 'ever', 'last', 'first', 'second',
        'something', 'anything', 'everything', 'nothing', 'someone',
        'anyone', 'everyone', 'thing', 'way', 'one', 'two', 'three',
        'people', 'person', 'man', 'woman', 'child', 'guy', 'lot',
        'part', 'place', 'case', 'point', 'group', 'problem', 'fact',
        'hand', 'eye', 'mr', 'mrs', 'question', 'number', 'kind',
        'reason', 'result', 'end', 'area', 'program', 'system',
        'example', 'issue', 'level', 'side', 'subject', 'type',        
        'post', 'comment', 'reply', 'thread', 'article', 'link',
        'please', 'thanks', 'thank', 'sorry', 'well', 'actually',
        'basically', 'literally', 'yeah', 'yes', 'hmm', 'huh',
        'wow', 'lol', 'hey', 'hi', 'hello', 'definitely',
        'totally', 'absolutely', 'exactly', 'probably', 'maybe',
        'honestly', 'seriously', 'obviously', 'clearly',
        'university', 'host', 'mail', 'bought', 'quick', 'fun',
        'life', 'said', 'going', 'really', 'much', 'back',
        'also', 'mean', 'even', 'love', 'many', 'refreshing',
        'powerful', 'inspiring', 'agree', 'bit', 'damn',
        'tried', 'trying', 'ended', 'spent', 'hit', 'made',
        'sharing', 'moment', 'deep', 'deeper', 'specific',
        'balance', 'difference', 'action', 'impact', 'hope',
        'curious', 'friend', 'fresh', 'might', 'got', 'getting'
    ])
    
    # Lemmatizer object to get word stems
    lemmatizer = WordNetLemmatizer()
    
    # Transform each post/comment into "bags of words"
    bow_list = []
    for text in content_list:
        if not text or not isinstance(text, str):
            continue
            
        tokens = word_tokenize(text.lower())  # tokenise
        tokens = [lemmatizer.lemmatize(t) for t in tokens]  # lemmatise
        tokens = [t for t in tokens if len(t) > 2]  # filter out words with less than 3 letters
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]  # filter out stopwords
        
        # If there's at least 1 word left, append to list
        if len(tokens) > 0:
            bow_list.append(tokens)
    
    print(f"Processed {len(bow_list)} documents for topic modeling")
    
    # Create dictionary and corpus
    dictionary = Dictionary(bow_list)
    
    # Filter words that appear less than 2 times or in more than 30% of posts
    dictionary.filter_extremes(no_below=2, no_above=0.3)
    
    corpus = [dictionary.doc2bow(tokens) for tokens in bow_list]
    
    print(f"Dictionary size: {len(dictionary)}")
    print(f"Corpus size: {len(corpus)}")
    
    # Train LDA model with 10 topics
    print("\nTraining LDA model with 10 topics...")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=10,
        random_state=42,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    
    # Calculate coherence score
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=bow_list,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    
    print(f"\nCoherence Score: {coherence_score:.4f}")
    print("Top 10 Topics:")
    
    # Display the topics
    for idx, topic in lda_model.print_topics(num_topics=10, num_words=10):
        print(f"\nTopic {idx + 1}:")
        print(topic)
    
    return lda_model, dictionary, corpus

def analyze_overall_sentiment(db_path='database.sqlite'):
    # Download VADER lexicon
    nltk.download('vader_lexicon', quiet=True)
    
    # Get all content
    content_list = get_all_content(db_path)
    print(f"\nAnalyzing sentiment for {len(content_list)} items...")
    
    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Calculate sentiment scores for each item
    sentiment_scores = []
    for text in content_list:
        if text and isinstance(text, str):
            score = sia.polarity_scores(text)['compound']
            sentiment_scores.append(score)
    
    # Create DataFrame
    df = pd.DataFrame({'sentiment_score': sentiment_scores})
    
    # Categorize sentiments
    df['sentiment_category'] = df['sentiment_score'].apply(
        lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
    )
    
    # Overall platform statistics
    print("Overall Platform Sentiment:")
    print(f"Total items analyzed: {len(df)}")
    print(f"Average sentiment score: {df['sentiment_score'].mean():.4f}")
    print(f"Median sentiment score: {df['sentiment_score'].median():.4f}")
    print(f"Standard deviation: {df['sentiment_score'].std():.4f}")
    
    print(f"\nSentiment Distribution:")
    sentiment_counts = df['sentiment_category'].value_counts()
    for category in ['Positive', 'Neutral', 'Negative']:
        count = sentiment_counts.get(category, 0)
        percentage = (count / len(df)) * 100
        print(f"  {category}: {count} ({percentage:.2f}%)")
    
    return df

def analyze_sentiment_by_topic(db_path='database.sqlite', lda_model=None, 
                                dictionary=None, corpus=None, df_sentiment=None):
    if lda_model is None or dictionary is None or corpus is None:
        print("Error: Need LDA model, dictionary, and corpus from topic modeling")
        return
    
    if df_sentiment is None:
        df_sentiment = analyze_overall_sentiment(db_path)
    
    print("\n\nSentiment Analysis by Topic:")
    
    # Assign dominant topic to each document
    topic_assignments = []
    for i, doc_bow in enumerate(corpus):
        doc_topics = lda_model.get_document_topics(doc_bow)
        if doc_topics:
            dominant_topic = max(doc_topics, key=lambda x: x[1])
            topic_assignments.append(dominant_topic[0])
        else:
            topic_assignments.append(-1)
    
    # Merge with sentiment data
    df_sentiment_with_topics = df_sentiment.head(len(topic_assignments)).copy()
    df_sentiment_with_topics['topic_id'] = topic_assignments
    
    # Filter out documents without topics
    df_sentiment_with_topics = df_sentiment_with_topics[df_sentiment_with_topics['topic_id'] != -1]
    
    # Calculate sentiment percentages for each topic
    for topic_id in sorted(df_sentiment_with_topics['topic_id'].unique()):
        topic_data = df_sentiment_with_topics[df_sentiment_with_topics['topic_id'] == topic_id]
        sentiment_dist = topic_data['sentiment_category'].value_counts()
        
        # Get topic keywords
        topic_words = lda_model.show_topic(topic_id, topn=5)
        words = [word for word, _ in topic_words]
        
        print(f"\nTopic {topic_id + 1} ({', '.join(words)}):")
        for category in ['Positive', 'Neutral', 'Negative']:
            count = sentiment_dist.get(category, 0)
            percentage = (count / len(topic_data)) * 100
            print(f"  {category}: {percentage:.1f}%")
    
    return df_sentiment_with_topics

if __name__ == "__main__":
    # Perform topic modeling
    lda_model, dictionary, corpus = top_10_topics()
    
    # Perform overall sentiment analysis
    df_sentiment = analyze_overall_sentiment()
    
    # Analyze sentiment by topic
    df_sentiment_by_topic = analyze_sentiment_by_topic(
        lda_model=lda_model,
        dictionary=dictionary,
        corpus=corpus,
        df_sentiment=df_sentiment
    )
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import sqlite3

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
    topics = []
    for idx, topic in lda_model.print_topics(num_topics=10, num_words=10):
        print(f"\nTopic {idx + 1}:")
        print(topic)

if __name__ == "__main__":
    top_10_topics()
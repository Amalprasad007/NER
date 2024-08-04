import requests
from bs4 import BeautifulSoup
import spacy
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

# Function to fetch news articles
def fetch_news_articles(api_key, num_articles=3):
    url = f'https://newsapi.org/v2/top-headlines?sources=bbc-news&apiKey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        selected_articles = []
        for article in articles[:num_articles]:
            content = article.get('content')
            if content and '[+' in content:
                # Fetch the full article text from the URL
                article_url = article.get('url')
                full_text = fetch_full_article_text(article_url)
                content = full_text if full_text else article.get('description')
            elif not content:
                content = article.get('description')
            selected_articles.append((article.get('title'), content))
        return selected_articles
    return []

# Function to fetch full article text from the URL
def fetch_full_article_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        full_text = ' '.join([para.get_text() for para in paragraphs])
        return full_text
    except Exception as e:
        print(f"Error fetching full article text: {e}")
        return None

# Function to extract first 5 sentences from the article
def get_first_five_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return ' '.join(sentences[:5])

# Function to extract entities using SpaCy
def extract_entities_spacy(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function to extract entities using NLTK
def extract_entities_nltk(text):
    nltk.download('punkt')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('averaged_perceptron_tagger')

    sentences = nltk.sent_tokenize(text)
    entities = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tags = nltk.pos_tag(words)
        tree = nltk.ne_chunk(tags, binary=False)
        for subtree in tree:
            if isinstance(subtree, nltk.Tree):
                entity = " ".join([word for word, tag in subtree.leaves()])
                entity_type = subtree.label()
                entities.append((entity, entity_type))
    return entities

# Compare the results from SpaCy and NLTK
def compare_ner(nltk_entities, spacy_entities):
    nltk_set = set(nltk_entities)
    spacy_set = set(spacy_entities)

    unique_to_nltk = nltk_set - spacy_set
    unique_to_spacy = spacy_set - nltk_set
    common_entities = nltk_set & spacy_set

    return unique_to_nltk, unique_to_spacy, common_entities

# Replace 'your_api_key' with your actual News API key
api_key = '7a8a5f8bb0b94f2f9de77d2751ecfac1'
articles = fetch_news_articles(api_key, num_articles=2)

for i, (title, article) in enumerate(articles):
    if article:
        first_five_sentences = get_first_five_sentences(article)
        print(f"\nArticle {i+1}:\nTitle: {title}\nContent: {first_five_sentences}")
        
        # Extract entities using NLTK
        nltk_entities = extract_entities_nltk(first_five_sentences)
        print(f"\nNamed Entities using NLTK for Article {i+1}:\n", nltk_entities)

        # Extract entities using SpaCy
        spacy_entities = extract_entities_spacy(first_five_sentences)
        print(f"\nNamed Entities using SpaCy for Article {i+1}:\n", spacy_entities)

        # Compare results
        unique_to_nltk, unique_to_spacy, common_entities = compare_ner(nltk_entities, spacy_entities)

        print(f"\nUnique to NLTK for Article {i+1}:\n", unique_to_nltk)
        print(f"\nUnique to SpaCy for Article {i+1}:\n", unique_to_spacy)
        print(f"\nCommon Entities for Article {i+1}:\n", common_entities)
    else:
        print(f"Article {i+1} content is empty or not available.")

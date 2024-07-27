import spacy
from nltk.corpus import wordnet as wn

# Map SpaCy tags to WordNet tags
def get_wordnet_pos(spacy_token):
    pos = spacy_token.pos_
    if pos == 'VERB':
        return wn.VERB
    elif pos == 'NOUN':
        return wn.NOUN
    elif pos == 'ADJ':
        return wn.ADJ
    elif pos == 'ADV':
        return wn.ADV
    return None

# Load synsets from WordNet based on word and its POS
def load_wordnet_synsets(word, pos = None):
    if pos:
        return wn.synsets(word, pos = pos)
    else:
        return wn.synsets(word)

# Check semantic relationships between two words, based on their WordNet synsets
def check_semantic_relationship(word1, word2, pos1 = None, pos2 = None):
    synsets1 = load_wordnet_synsets(word1, pos = pos1)
    synsets2 = load_wordnet_synsets(word2, pos = pos2)
    
    for syn1 in synsets1:
        for syn2 in synsets2:
            if syn1 == syn2:
                return True
            if syn2 in syn1.hypernyms() or syn1 in syn2.hypernyms():
                return True
    return False

# Highlight words in text_B that are semantically related to key words in text_A
def highlight_texts(text_A, text_B, spacy_nlp, mark):
    doc_A = spacy_nlp(text_A)
    doc_B = spacy_nlp(text_B)
    ignore_verbs = {"be", "am", "is", "are", "was", "were", "being", "been", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can", "could", "may", "might", "must"}
    important_deps = {'nsubj', 'dobj', 'pobj', 'attr', 'ROOT', 'compound'}
    
    key_words = [(token.text, get_wordnet_pos(token)) for token in doc_A if token.dep_ in important_deps]

    highlighted_text_B = []
    for token in doc_B:
        token_text = token.text
        token_pos = get_wordnet_pos(token)
        if token.lemma_ not in ignore_verbs and any(check_semantic_relationship(key_word, token.text, pos1=pos, pos2=token_pos) for key_word, pos in key_words):
            token_text = f"{mark}{token_text}{mark}"
        highlighted_text_B.append(token_text)

    return ' '.join(highlighted_text_B)
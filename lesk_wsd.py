import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from functools import reduce


def lesk_extended(index_word, context):
    tagged_context = preprocessing(context)
    word_pos = tagged_context[index_word]
    wnl = WordNetLemmatizer()

    # If word is not tagged
    if pos_map_nltk_wn(word_pos[1]) is '':
        return None

    lemma_pos = (wnl.lemmatize(word_pos[0], pos_map_nltk_wn(word_pos[1])), pos_map_nltk_wn(word_pos[1]))
    filtered_tagged_context = preprocessing2(tagged_context)
    lemmatized_tagged_context = [(wnl.lemmatize(w, pos), pos) for (w, pos) in filtered_tagged_context]
    return lesk(lemma_pos, set(lemmatized_tagged_context))


def lesk(word_to_dis, context):
    synsets = wn.synsets(word_to_dis[0], word_to_dis[1])
    best_syn = None
    best_score = 0.0
    for ss in synsets:
        hierarchical_synset = set(
            ss.substance_meronyms() + ss.hypernyms() + ss.hyponyms() + ss.member_holonyms() + ss.part_meronyms()
        )

        value = list(map(lambda x: set(preprocessing(x.definition())), hierarchical_synset))
        scores = list(map(lambda x: len(context.intersection(x))/max(len(context), len(x)), value))
        hierarchical_score = reduce((lambda x, y: x + y), scores, 0)

        gloss = ss.definition()
        gloss = set(preprocessing(gloss))
        total_score = len(context.intersection(gloss))/max(len(context), len(gloss)) + hierarchical_score
        if total_score > best_score:
            best_syn = ss
            best_score = total_score
    if best_score == 0.0 and len(synsets) > 0:
            best_syn = synsets[0]
    return best_syn


def preprocessing(sentence):
    tokenized_context = word_tokenize(sentence)  # list string
    return nltk.pos_tag(tokenized_context)  # list(tuple(string, string))


def preprocessing2(tokenized_context_tag):
    return [(w, pos_map_nltk_wn(pos)) for w, pos in tokenized_context_tag if (pos_map_nltk_wn(pos) != '')]


def pos_map_nltk_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return ''


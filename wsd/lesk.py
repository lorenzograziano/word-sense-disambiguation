from nltk.corpus import stopwords, wordnet


def lesk_without_stop_words(context_sentence, ambiguous_word, pos=None, lang='english'):
    # Attempt to make lesk more accurate removing stop words
    stop_words = set(stopwords.words(lang))
    context = set(context_sentence)
    synsets = wordnet.synsets(ambiguous_word)

    if pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == pos]
    if not synsets:
        return None
    _, sense = max(
        (len(context.intersection(set(ss.definition().split()) - stop_words)), ss) for ss in synsets
    )
    return sense

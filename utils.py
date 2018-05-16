from nltk import tree
from wsd.lesk import lesk_without_stop_words
from nltk.wsd import lesk


def retrieve_sent(ambiguos_word, all_tagged_sents):
    # This method retrieve all the sentences containing the specified ambiguous word
    # and the corresponding synset from a tagged corpora structured as list of sentences.
    # Each sentence is represented as a list of tagged chunks (in tree form).
    saved_tagged_sent = []
    saved_synset = []

    for tagged_sent in all_tagged_sents:
        sent_to_save = False
        sent_words = []
        for tagged_chunk in tagged_sent:
            synset = ""
            if isinstance(tagged_chunk, tree.Tree):
                current_word = tagged_chunk.leaves()
                sent_words = sent_words + current_word
                synset = tagged_chunk.label()
            else:
                current_word = tagged_chunk
                sent_words = sent_words + current_word
                if len(tagged_chunk) == 2:
                    synset = tagged_chunk[1]
            if ambiguos_word in current_word:
                sent_to_save = True
                saved_synset.append(synset)
        if sent_to_save:
            saved_tagged_sent.append(sent_words)
    return saved_tagged_sent, saved_synset


def retrieve_lesk_senses(ambiguos_word, sent_with_disambigued_word):
    lesk1_syns = []
    lesk2_syns = []

    i = 0
    correct_syns_lesk1 = 0
    correct_syns_lesk2 = 0

    retrieved_make_sent = sent_with_disambigued_word[0]
    for sent in retrieved_make_sent:
        lesk1 = lesk(sent, ambiguos_word)
        lesk1_syns.append(lesk1)
        lesk2 = lesk_without_stop_words(sent, ambiguos_word)
        lesk2_syns.append(lesk2)
        # print(retrievedMake[1][i])
        # print(lesk1.lemmas())
        if sent_with_disambigued_word[1][i] == lesk1.lemmas():
            correct_syns_lesk1 = correct_syns_lesk1 + 1
        if sent_with_disambigued_word[1][i] == lesk2.lemmas():
            correct_syns_lesk2 = correct_syns_lesk2 + 1
    return lesk1_syns, lesk2_syns

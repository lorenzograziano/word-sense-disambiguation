import lesk_wsd
import nltk.tokenize as tokenize


def mainWSDTask(input_file):

    # mainWSDTAsk is the main method; it takes as input an absolute path to the json dictionary containing
    # definitions to be disambiguated. It returns a dictionary containing key: List(List(couple(word, sense))). A list
    # for each definition and a list for each sentence of the definition

    with open(input_file, 'r') as input:
        text = input.readlines()

    keys = []
    text_split = []
    for line in text:
        if ":" in line:
            key_descriptor = line.split(":", 1)
            keys.append(key_descriptor[0].strip().replace("\"", ""))
            text_split.append(key_descriptor[1].strip().replace("\"", ""))

    list_list_sent = list()
    for definition in text_split:
        sentences = tokenize.sent_tokenize(definition)
        list_list_sent.append(sentences)

    disambiguated_definition = dict()

    for i, definition in enumerate(list_list_sent):
        disambiguated_sents = []
        for sent in definition:
            wsd_sent = []
            tokenized_sent = tokenize.word_tokenize(sent)
            for j in range(0, len(tokenized_sent)):
                wsd_sent.append((tokenized_sent[j], lesk_wsd.lesk_extended(j, sent)))
            disambiguated_sents.append(wsd_sent)
        disambiguated_definition[keys[i]] = disambiguated_sents

    return disambiguated_definition
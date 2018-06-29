import lesk_wsd
import nltk.tokenize as tokenize


input_file = '/home/lorenzo/PycharmProjects/wsd/definizioni.json'

with open(input_file, 'r') as input:
    text = input.readlines()


keys = []
text_split = []     #List of definitions
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
        for i in range(0, len(tokenized_sent)):
            wsd_sent.append((tokenized_sent[i], lesk_wsd.lesk_extended(i, sent)))
        disambiguated_sents.append(wsd_sent)
    disambiguated_definition[keys[i]] = disambiguated_sents
print(disambiguated_definition)
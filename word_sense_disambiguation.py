from nltk.corpus import semcor
import utils


all_tagged_sents = semcor.tagged_sents(tag='sem')

# retrieve_sentences for chosen word to disambiguate
retrievedBreak = utils.retrieve_sent("break", all_tagged_sents)
retrievedMake = utils.retrieve_sent("make", all_tagged_sents)
retrievedRun = utils.retrieve_sent("run", all_tagged_sents)

# print(len(retrievedMake[0]))
# print(len(retrievedBreak[0]))
# print(len(retrievedRun[0]))


# retrieve Senses for chosen word with two different version of lesk
lesk1_make, lesk2_make = utils.retrieve_lesk_senses("make", retrievedMake[0])


print(lesk1_make)
print("______________________________________")
print(lesk2_make)

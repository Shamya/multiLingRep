import ujson
from random import shuffle
from libraries.acs import acs,acs_map

def transform(sentences, language):
    tcnt = 0
    for sentence in sentences:
        for ind in xrange(len(sentence)):
            word = acs_map(sentence[ind], language)
            if(word != sentence[ind]):
                tcnt+=1
            sentence[ind] = word
    print "Total Transformations", language,  " Happened is ", str(tcnt)
    return sentences

# f = open('combined_data.txt', 'w')
import codecs
f1 = codecs.open("acs_combined_en_it_fr_es_data.txt", "w", "utf-8")
 
ENG_sentences = []
with open('data/engdata.json', 'r') as f:
    ENG_sentences = ujson.load(f)

ENG_sentences = transform(ENG_sentences, language='english')
sentences = ENG_sentences
print len(sentences)
for line in sentences:
    t = ' '.join(line)
    f1.write(t + "\n")

del ENG_sentences
del sentences

ESP_sentences = []
with open('data/espdata.json', 'r') as f:
    ESP_sentences = ujson.load(f)

ESP_sentences = transform(ESP_sentences, language='spanish')
sentences = ESP_sentences
print len(sentences)
for line in sentences:
    t = ' '.join(line)
    f1.write(t + "\n")

del ESP_sentences
del sentences

ITA_sentences = []
with open('data/itadata.json', 'r') as f:
    ITA_sentences = ujson.load(f)

ITA_sentences = transform(ITA_sentences, language='italian')
sentences = ITA_sentences
print len(sentences)
for line in sentences:
    t = ' '.join(line)
    f1.write(t + "\n")

del ITA_sentences
del sentences


FRE_sentences = []
with open('data/fredata.json', 'r') as f:
    FRE_sentences = ujson.load(f)


FRE_sentences = transform(FRE_sentences, language='french')
sentences = FRE_sentences
print len(sentences)
for line in sentences:
    t = ' '.join(line)
    f1.write(t + "\n")

del FRE_sentences
del sentences

f1.close()
from fileUtilities import load_pkl_file, save_pkl_file
import io
from collections import defaultdict

LANGUAGES = ["english", "spanish"]
DATA = "../omega_wiki_map/"

def loadIntoOmegaWikiConceptMap():
  for language in LANGUAGES:
    language_map_concept = defaultdict(list)
    filename = language + "_words.txt"
    f = io.open(DATA+filename)
    for line in f.readlines():
      tokens = line.split("\t")
      assert len(tokens)==2
      language_map_concept[tokens[0]].append(tokens[1])
    save_pkl_file(language + "_map_concept.txt", language_map_concept)
    concept_map_language = defaultdict(list)
    filename = "concept_" + language + "words.txt"
    f = io.open(DATA+filename)
    for line in f.readlines():
      tokens = line.split("\t")
      assert len(tokens)==2
      assert tokens[0] not in concept_map_language
      concept_map_language[tokens[0]] = (tokens[1].split(","))
    save_pkl_file("concept_map_" + language , concept_map_language)
    

loadIntoOmegaWikiConceptMap()

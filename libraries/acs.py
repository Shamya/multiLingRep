import random
from OmegaWiki import OmegaWiki
from libraries.fileUtilities import load_pkl_file
import time
import random

def acs(word):
  # split(document)
  v = random.random()
  # print v
  if(v>0.5):
    om = OmegaWiki(word,"English")
    # Choose a concept
    if(len(om.Dict.keys()) > 0):
      key = random.choice(om.Dict.keys())
      time.sleep(0.1)
      # Choose a language
      translated = random.choice(om.Dict[key].keys())
      return om.Dict[key][translated]
    else:
      return word

  return word

key_Lang = {1 : "english",
 2 : "spanish"
}

Lang_key = { "english" : 1,
 "spanish" : 2
}

english_map_concept = load_pkl_file("libraries/english_map_concept.txt")
spanish_map_concept = load_pkl_file("libraries/spanish_map_concept.txt")
concept_map_english = load_pkl_file("libraries/concept_map_english")
concept_map_spanish = load_pkl_file("libraries/concept_map_spanish")

def model(ignore):
  keys = [1,2]
  keys.remove(ignore)
  random.choice(keys)
  

def acs_map(word, language='english', other_language='spanish'):
  v = random.random()
  # print v
  if(v>0.0001):
    # om = OmegaWiki(word,"English")
    # Choose a concept
    if(language == "english"):
      langConcept = english_map_concept
      conceptOtherLang = concept_map_spanish
    elif(language == "spanish"):
      langConcept = spanish_map_concept
      conceptOtherLang = concept_map_english
    if(word in langConcept):
      conceptKey = random.choice(langConcept[word]).strip()
      if(len(conceptOtherLang[conceptKey]) > 0):
        transformedWord = random.choice(conceptOtherLang[conceptKey]).strip()
        print word, " transformed to ", transformedWord
        return transformedWord #random.choice(conceptOtherLang[conceptKey]).strip()
  return word  



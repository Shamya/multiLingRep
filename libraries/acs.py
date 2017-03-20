import random
from OmegaWiki import OmegaWiki

def acs(word):
  # split(document)
  v = random.random()
  print v
  if(v>=0.5):
    om = OmegaWiki(word,"English")
    # Choose a concept
    # print om.Dict
    key = random.choice(om.Dict.keys())
    # print key
    # Choose a language
    translated = random.choice(om.Dict[key].keys())
    # print translated
    return om.Dict[key][translated]

    # om.displayLang()
    # print om.Dict
    # pprint(om.Dict)
    # pprint(om.RevDict)
    # pprint(om.Lang)
  return word


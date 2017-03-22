import random
from OmegaWiki import OmegaWiki
import time
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


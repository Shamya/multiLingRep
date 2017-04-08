import ujson
def spanishcorpus(mode="Load"):
    # Global list variable for Spanish sentences after processing
  ESP_sentences = []
  if mode=="Load":
    # Reading data back
    print "Loading Spanish"
    with open('data/espdata.json', 'r') as f:
         ESP_sentences = ujson.load(f)
    f.close()
    print "Spanish Loaded"   
  print "Done with Spanish"
  return ESP_sentences

def englishcorpus(mode="Load"):
    # Global list variable for Spanish sentences after processing
  ESP_sentences = []
  if mode=="Load":
    # Reading data back
    print "Loading English"
    with open('data/engdata.json', 'r') as f:
         ESP_sentences = ujson.load(f)
    f.close()
    print "English Loaded"   
  print "Done with English"
  return ESP_sentences
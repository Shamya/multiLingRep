import json, requests, re
from collections import defaultdict

def rec_dd():
   return defaultdict(rec_dd)

def parseKeyForMeaningId(key):
   match = re.search('([0-9]+)', key)
   if match:
      MeaningID = match.group(1)
   else:
      assert False, "Error raised"
   return MeaningID

def parseKeyForLanguage(key):
   match = re.search('Defined Meaning ([0-9]+) ([A-Za-z]+)', key)
   if match:
      Language = match.group(2)
   else:
      assert False, "Error raised"
   print Language
   return Language



class OmegaWiki:
   'OmegaWiki For Translations'
   
   def __init__(self, word, language):

      url = 'http://localhost:8080/mapping'
      params = dict(
          name=word,
          language=language
      )

      resp = requests.get(url=url, params=params)
      data = json.loads(resp.text)
      self.word = word
      self.language = language
      self.data = data
      self.parseData(self.data)

   def parseData(self, data):
      self.Dict = rec_dd()
      print data
      for key in data:
         if(re.search('Defined Meaning', key)):
            MeaningID = parseKeyForMeaningId(key) # The integer Key Denoting the Value
            Language = parseKeyForLanguage(key) # The String Key Denoting the Language
            self.Dict[MeaningID][Language] = data[key]



   def displayLang(self):
      print "DISPLAY"
      for key in self.Dict:
         print "Translation for "
         for innerMapKey in self.Dict[key]:
            print innerMapKey + "    " + self.Dict[key][innerMapKey]

# oq = OmegaWiki("blue","English")
# oq.parseData(oq.data)
# print oq.Dict

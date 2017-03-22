from libraries.OmegaWiki import OmegaWiki
from libraries.parser import returnDataSets
from libraries.dataSetUtility import splitDataSets, prepareDataSetForSKlearn
from libraries.acs import acs
from libraries.embeddings import process_and_retrieve_model
def main():
  # Datasets = returnDataSets()
  # Train, Valid, Test = splitDataSets(Datasets)
  # prepareDataSetForSKlearn(Train,Valid,Test)
  # The following Omega Wiki requires that you run your server in localhost
  # line = u"anarchism is a political philosophy that advocates self-governed societies based on voluntary institution"
  # # words = ["anarchism", "is", "a", "political", "philosophy"]
  # words = line.split()
  # print words
  # result = []
  # for word in words:
  #   result.append(acs(word))
  # print result
  model = process_and_retrieve_model()

  # om = OmegaWiki("blue","English")
  # om.displayLang()


if __name__ == "__main__":
    main()

from libraries.OmegaWiki import OmegaWiki
from libraries.parser import returnDataSets
from libraries.dataSetUtility import splitDataSets, prepareDataSetForSKlearn
def main():
  Datasets = returnDataSets()
  Train, Valid, Test = splitDataSets(Datasets)
  prepareDataSetForSKlearn(Train,Valid,Test)
  # The following Omega Wiki requires that you run your server in localhost
  # om = OmegaWiki("blue","English")
  # om.displayLang()


if __name__ == "__main__":
    main()

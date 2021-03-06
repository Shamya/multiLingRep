# from libraries.OmegaWiki import OmegaWiki
from libraries.parser import returnDataSets
from libraries.dataSetUtility import splitDataSets, prepareDataSetForSKlearn, prepareDataSetForSKlearnWithEmbeddings
# from libraries.acs import acs,acs_map
# from libraries.embeddings import process_and_retrieve_model, gather_data
# from libraries.cbow import generate_embeddings
from libraries.sentiment_classifier import sentimentAnalysisSpanishDataset,sentimentAnalysisEnglishDataset
# from libraries.corpus import spanishcorpus, englishcorpus
def main():
  # Logistic Regression for Bag Of Words
  # Datasets = returnDataSets()
  # Train, Valid, Test = splitDataSets(Datasets)
  # prepareDataSetForSKlearn(Train,Valid,Test)

  # print acs_map("haze", language='english', other_language='spanish')

  sentimentAnalysisSpanishDataset()
  # sentimentAnalysisEnglishDataset()
  # Datasets = returnDataSets()
  # Train, Valid, Test = splitDataSets(Datasets)
  # #prepareDataSetForSKlearnprepareDataSetForSKlearn(Train,Valid,Test)
  # prepareDataSetForSKlearnWithEmbeddings(Train, Valid, Test)


  # THE COMMENTED ONE USES GENSIM
  # model = process_and_retrieve_model()

  # Using Tensorflow
  # multilingual_data = gather_data()
  # embeddings, dictionary, rev_dictionary = generate_embeddings(englishcorpus())


if __name__ == "__main__":
    main()

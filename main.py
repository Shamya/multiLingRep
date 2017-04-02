from libraries.OmegaWiki import OmegaWiki
from libraries.parser import returnDataSets
from libraries.dataSetUtility import splitDataSets, prepareDataSetForSKlearn, prepareDataSetForSKlearnWithEmbeddings
from libraries.acs import acs
from libraries.embeddings import process_and_retrieve_model, gather_data
from libraries.cbow import generate_embeddings
from libraries.sentiment_classifier import sentimentAnalysisFrenchDataset
def main():
  # Logistic Regression for Bag Of Words
  # Datasets = returnDataSets()
  # Train, Valid, Test = splitDataSets(Datasets)
  # prepareDataSetForSKlearn(Train,Valid,Test)
  sentimentAnalysisFrenchDataset()
  # Datasets = returnDataSets()
  # Train, Valid, Test = splitDataSets(Datasets)
  # #prepareDataSetForSKlearn(Train,Valid,Test)
  # prepareDataSetForSKlearnWithEmbeddings(Train, Valid, Test)


  # THE COMMENTED ONE USES GENSIM
  # model = process_and_retrieve_model()

  # Using Tensorflow
  # multilingual_data = gather_data()
  # embeddings, dictionary, rev_dictionary = generate_embeddings(multilingual_data)


if __name__ == "__main__":
    main()

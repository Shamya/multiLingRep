import pickle
import codecs
import io
def save_pkl_file(filename, data):
  file_handle = open(filename,mode='w')
  pickle.dump( data, file_handle )

def load_pkl_file(filename):
  file_handle = open(filename,mode='r')
  return pickle.load( file_handle )
f = open('vectors_cbow_combined_mul.txt', 'r')
l = f.readline()
x = {}
import pickle
for line in f:
    tokens = line.split()
    assert len(tokens) == 301
    x[tokens[0]] = [float(t) for t in tokens[1:]]


def save_pkl_file(filename, data):
  file_handle = open(filename,mode='w')
  pickle.dump( data, file_handle )

import pickle
# save_pkl_file("en_fr_it_es_word_vectors.txt", data)


file_handle = open("en_fr_it_es_word_vectors_com.txt",mode='w')
pickle.dump(x, file_handle )
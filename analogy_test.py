import vecto
import vecto.embeddings
from vecto.utils.data import save_json
import pandas
from pandas.io.json import json_normalize
from matplotlib import pyplot as plt
from vecto.benchmarks.analogy import LRCos


class embeddings:
    name = 0
    def __init__(self, dict):
        i = dict['path']
        self.embedding_name = i.split('/')[5]
        self.embedding_directory = i
        self.embeddings = vecto.embeddings.load_from_dir(self.embedding_directory)
        self.citation = dict['citation']
        self.description = dict['description']

    def get_analogy(self, dataset):
        self.dataset_path = dataset
        analogy = LRCos()
        self.result = analogy.get_result(self.embeddings, dataset)
        save_json(self.result, "./res.json")
        return self.result

    def get_row(self):
        self.row = {}
        for subcategory in self.subcategories:
            self.row[subcategory] = 1

    def get_dictionary(self):
        dictionary = {}
        dictionary['embedding'] = self.embedding_name
        for i in self.result:
            subcategory = i["experiment_setup"]["subcategory"]
            results = i['result']
            missing_pairs = self.get_length(subcategory) - i["experiment_setup"]["cnt_questions_total"]
            dictionary[subcategory] = {'results' : results,'missing pairs' : missing_pairs }
        dictionary['description'] = self.description
        dictionary['citation'] = self.citation
        print(dictionary)
        return dictionary

    def get_length(self, filename):
        i = 0
        file = open(self.dataset_path+ filename, 'r')
        for line in file:
            i+=1
        return i
#embeddigns = vecto.embeddings.load_from_dir("/home/downey/PycharmProjects/jupyter_example/wiki.got(1)/")
#embeddigns = vecto.embeddings.load_from_dir("/home/downey/PycharmProjects/jupyter_example/glove.twitter.27B/")
#embeddigns = vecto.embeddings.load_from_dir("/home/downey/PycharmProjects/jupyter_example/!Demo2/embeddings/bnc/") #works
#embeddigns = vecto.embeddings.load_from_dir("/home/downey/PycharmProjects/jupyter_example/glove.6b/glove.6B/")      #works
#embeddigns = vecto.embeddings.load_from_dir("/home/downey/PycharmProjects/jupyter_example/glove.42B/")
#embeddigns = vecto.embeddings.load_from_dir("/home/downey/PycharmProjects/jupyter_example/enwik/")
#embeddigns = vecto.embeddings.load_from_dir("/home/downey/PycharmProjects/jupyter_example/training-monolingual/")
#embeddigns = vecto.embeddings.load_from_dir("/home/downey/PycharmProjects/jupyter_example/lstm/1/word/") #works

directories = [{'path':"/home/downey/PycharmProjects/jupyter_example/!Demo2/embeddings/bnc/",
                'citation':'',
                'description':''},
               {'path':"/home/downey/PycharmProjects/jupyter_example/lstm/1/word/",
                'citation':'',
                'description':'subword model with LSTM composition function from Li et al. trained on text8 '},
               {'path':"/home/downey/PycharmProjects/jupyter_example/glove.6b/glove.6B/",
                'citation':'Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: '
                           'Global Vectors for Word Representation. [pdf] [bib] ',
                'description':'Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d'
                              ' vectors)'}]



df = pandas.DataFrame()
dicts = []
for file in directories:
    embedding = embeddings(file)
    #results = embedding.get_analogy("/home/downey/PycharmProjects/jupyter_example/!Demo2/datasets/bats_small/")
    results = embedding.get_analogy("/home/downey/PycharmProjects/jupyter_example/BATS_3.0/")
    dicts.append(embedding.get_dictionary())
    df = pandas.DataFrame(dicts).set_index('embedding')
    df.to_csv('table1.csv')
    print(df)

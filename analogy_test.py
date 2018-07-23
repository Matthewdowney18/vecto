import vecto
import vecto.embeddings
from vecto.utils.data import save_json
import pandas
from vecto.benchmarks.analogy import LRCos


class embeddings:
    name = 0
    def __init__(self, path):
        i = path
        self.embedding_name = i.split('/')[5]
        self.embedding_directory = i
        self.embeddings = vecto.embeddings.load_from_dir(self.embedding_directory)

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
            dictionary[subcategory+' results'] = results
            dictionary[subcategory + ' missing pairs'] = missing_pairs
            dictionary['citation'] = self.get_citation()
            dictionary['description'] = self.get_description()
        print(dictionary)
        return dictionary


    def get_citation(self):
        citation = self.result[0]['experiment_setup']['embeddings']['vocabulary']['corpus']['bib']
        return citation


    def get_description(self):
        description = ''
        if 'description' in self.result[0]['experiment_setup']['embeddings']:
            description += self.result[0]['experiment_setup']['embeddings']['description']+ ", "
            description += "corpus name: "+self.result[0]['experiment_setup']['embeddings']['vocabulary']['corpus']['name']
            description += ", language: "+self.result[0]['experiment_setup']['embeddings']['vocabulary']['corpus']['language']
            description += ", size: "+str(self.result[0]['experiment_setup']['embeddings']['vocabulary']['corpus']['size'])
            #description += ", size: " + self.result[0]['experiment setup']['embeddings']['vocabulary']['corpus']['size']
        return description


    def get_length(self, filename):
        i = 0
        file = open(self.dataset_path + filename, 'r')
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

directories = ["/home/downey/PycharmProjects/jupyter_example/!Demo2/embeddings/bnc/",
               "/home/downey/PycharmProjects/jupyter_example/lstm/1/word/",
               "/home/downey/PycharmProjects/jupyter_example/glove.6b/glove.6B/"]




df = pandas.DataFrame()
dicts = []
for file in directories:
    embedding = embeddings(file)
    results = embedding.get_analogy("/home/downey/PycharmProjects/jupyter_example/!Demo2/datasets/bats_small/")
    dicts.append(embedding.get_dictionary())
    df = pandas.DataFrame(dicts).set_index('embedding')
    df.to_csv('table1.csv')

    print(df)

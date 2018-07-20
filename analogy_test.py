import vecto
import vecto.embeddings
from vecto.utils.data import save_json
import pandas
from pandas.io.json import json_normalize
from matplotlib import pyplot as plt
from vecto.benchmarks.analogy import LRCos


class embeddings:
    def __init__(self, i):
        self.embedding = i.split('/')[5]
        self.embedding_directory = i
        self.embeddings = vecto.embeddings.load_from_dir(self.embedding_directory)

    def get_analogy(self, dataset):
        analogy = LRCos()
        self.result = analogy.get_result(self.embeddings, dataset)
        save_json(self.result, "./res.json")
        return self.result

    def get_row(self):
        self.row = {}
        for subcategory in self.subcategories:
            self.row[subcategory] = 1

    def get_missing_pairs(self):
        self.missing_pairs=[]
        for i in results:
            self.missing_pairs.append(i)

    def get_subcategory_results(self):
        df = pandas.concat([self.json_to_df(j) for j in results])
        df.to_csv('dataframe2.csv')
        self.get_table(df)

    def json_to_df(self, results):
        meta = [["experiment_setup", "subcategory"], ["experiment_setup", "method"]]
        df = json_normalize(results, record_path=["details"], meta=meta)
        df["reciprocal_rank"] = 1 / (df["rank"] + 1)
        df["embedding"] = self.embedding
        df.to_csv('dataframe.csv')
        return df

    def get_table(self, df):
        group = df.groupby(["experiment_setup.subcategory", "experiment_setup.method", "embedding"])
        means = group.mean()
        print(means)
        means.to_csv('dataframe3.csv')
        means.reset_index(inplace=True)
        means = means.loc[:, ["experiment_setup.subcategory", "experiment_setup.method", "reciprocal_rank", "embedding"]]
        unstacked = means.groupby(['experiment_setup.subcategory', 'experiment_setup.method'])[
            'reciprocal_rank'].aggregate('first').unstack()
        self.dataframe = means
        unstacked.plot(kind="bar")
        plt.show()


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

for file in directories:
    embedding = embeddings(file)
    results = embedding.get_analogy("/home/downey/PycharmProjects/jupyter_example/!Demo2/datasets/bats_small/")
    embedding.get_subcategory_results()
    df2 = embedding.dataframe.set_index("experiment_setup.subcategory")
    df.append(df2)
    df2.to_csv('dataframe5.csv')
    print(df2.groupby('embedding', axis=1))

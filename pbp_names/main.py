import pandas as pd
import numpy as np
import os
import spacy
import random
import time

class Main:
    def __init__(self, _dir):
        self._dir = _dir
        self.data_dir = self._dir + "/data/"
        self.models_dir = self._dir + "/models/"
        self.pbp_dir = self._dir + "../playByPlay_v2/data/clean/"
        self.nlp = spacy.load(self.models_dir + "names_model")
        return
    def saveFrame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    def printProgressBar(self, iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()
        return
    def build_all_details(self):
        fns = os.listdir(self.pbp_dir)
        file = open((self.data_dir + "all_details.txt"), "w")
        for index, fn in enumerate(fns):
            self.printProgressBar(index, len(fns), "Building all_details")
            df = pd.read_csv(self.pbp_dir + fn)
            df.dropna(subset=['detail'], inplace=True)
            details = '\n'.join(df['detail'].values)
            file.write(details + "\n")
        file.close()
        return
    def test_names(self):
        file = open((self.data_dir + "all_details.txt"), "r")
        lines = (file.read()).split("\n")
        lines = random.sample(lines, k=int(len(lines)*0.1))
        for sentence in lines:
            doc = self.nlp(sentence)
            f_names = list(set([ent.text for ent in doc.ents if ent.label_ == "PERSON"]))
            if len(f_names) == 0 and 'coin toss' not in sentence and 'Overtime' not in sentence and 'challenged' not in sentence:
                print(sentence)
        return
    def get_names(self, sentence: str):
        doc = self.nlp(sentence)
        return list(set([ent.text for ent in doc.ents if ent.label_ == "PERSON"]))
    
# END / Main

##################################

# m = Main("./")

# m.test_names()
import pandas as pd
import numpy as np
import os

class TestFeature:
    def __init__(self):
        self.data_dir = os.getcwd() + "/features/data/"
        return
    def build(self):
        data = {
            'id': [1, 2, 3],
            'col': ['a', 'b', 'c']
        }
        df = pd.DataFrame(data)
        df.to_csv("%s.csv" % (self.data_dir + "test_feature"))
        return
    
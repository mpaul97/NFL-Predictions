import os
import pandas as pd

class SomeFeature:
    def __init__(self):
        self._dir: str = os.getcwd()[:os.getcwd().index("main_v2")+len("main_v2")]+"/"
        self.data_dir = self._dir + "games/features/data/"
        return
    def some_feature_build_func(self):
        data = {
            "A": [1, 2, 3], "B": [4, 5, 6]
        }
        df = pd.DataFrame(data=data)
        df.to_csv("%s.csv" % (self.data_dir + "test_feature"))
        return

#######################    

# SomeFeature().some_feature_build_func()
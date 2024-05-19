import os

try:
    from features.test_feature import SomeFeature
except ModuleNotFoundError:
    from games.features.test_feature import SomeFeature

class Build:
    def __init__(self):
        self._dir: str = os.getcwd()[:os.getcwd().index("main_v2")+len("main_v2")]+"/"
        self.data_dir = self._dir + "games/features/data/"
        return
    def run(self):
        SomeFeature().some_feature_build_func()
        return

##############################
    
Build().run()

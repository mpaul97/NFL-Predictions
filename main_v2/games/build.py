import os

try:
    from features.test_feature import SomeFeature
except ModuleNotFoundError:
    from games.features.test_feature import SomeFeature

class SomeClass:
    def __init__(self):
        self.k_dir: str = os.getcwd()
        print(os.listdir(self.k_dir))
        return
    def foo(self):
        SomeFeature().some_feature_build_func()
        return
    
############################
    
SomeClass()


import os

from games.build import SomeClass

class Main:
    def __init__(self):
        
        return
    def run_all(self):
        SomeClass().foo()
        return
    
#######################

if __name__ == '__main__':
    m = Main()
    m.run_all()
    print(os.listdir(os.getenv("MAIN_GAMES_FEATURES_DATA_DIR")))
import os
import sys
sys.path.append("../")

from paths import ALL_PATHS

from games.build import Build as GamesBuild

class Main:
    def __init__(self):
        
        return
    def run_all(self):
        GamesBuild(paths=ALL_PATHS).run()
        return
    
#######################

if __name__ == '__main__':
    m = Main()
    m.run_all()
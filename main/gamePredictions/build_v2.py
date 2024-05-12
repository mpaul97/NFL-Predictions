import pandas as pd
import numpy as np
import os
import sys

from features.test_feature import TestFeature

class Build:
    def __init__(self):
        self.features = [
            TestFeature()
        ]
        return
    def main(self):
        for feature in self.features:
            feature.build()
        return
    
################################
    
b = Build()

b.main()
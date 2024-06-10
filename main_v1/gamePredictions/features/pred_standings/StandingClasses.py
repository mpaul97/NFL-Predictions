import pandas as pd
import numpy as np

class Standing:
    def __init__(self, abbr, year):
        self.abbr = abbr
        self.year = year
        self.wins = 0
        self.loses = 0
        self.ties = 0
        self.wl = 0
        self.confWins = 0
        self.confLoses = 0
        self.confTies = 0
        self.confWl = 0
        self.divWins = 0
        self.divLoses = 0
        self.divTies = 0
        self.divWl = 0
        self.sov = 0 # strength of victory -> combined win lose ratio of teams defeated
        self.sos = 0 # strength of schedule -> combined win lose ratio of all teams played against
        self.pointsFor = 0
        self.pointsAgainst = 0
        self.tdsFor = 0
        self.tdsAgainst = 0
        self.division = ''
        self.conference = ''

    def getObjKey(self):
        return self.abbr + "-" + str(self.year)

    def asDict(self):
        return vars(self)

    def zeroDivision(self, n, d):
        return n/d if d else 0

    def wlFunc(self, wins, loses, ties):
        return round(self.zeroDivision(((2 * wins) + ties), (2 * (wins + loses + ties))), 4)

    def updateWlRatios(self):
        self.wl = self.wlFunc(self.wins, self.loses, self.ties)
        self.confWl = self.wlFunc(self.confWins, self.confLoses, self.confTies)
        self.divWl = self.wlFunc(self.divWins, self.divLoses, self.divTies)

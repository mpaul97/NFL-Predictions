import spacy
import os
import regex as re
import numpy as np

nlp = spacy.load("D:/NFLPredictions3/pbp_custom_ners/models/model-best")

class Entity:
    def __init__(self, text: str, label: str, start: int, end: int):
        self.text = text
        self.label = label
        self.start = start
        self.end = end
        return
    def show(self):
        print(self.__dict__)
        return

ALL_ENTS = [
    'PASSER', 'RECEIVER', 'RUSHER', 'TACKLER',
    'DEFENDER', 'PENALIZER', 'FUMBLER', 
    'FUMBLE_RECOVERER', 'FUMBLE_FORCER',
    'INTERCEPTOR', 'KICKER', 'PUNTER', 
    'RETURNER', 'SACKER', 'LATERALER',
    'OTHER', 'TEAM_NAME', 'BLOCKER'
]

PID_ENTS = [
    'pid_PASSER', 'pid_RECEIVER', 'pid_RUSHER', 'pid_TACKLER',
    'pid_DEFENDER', 'pid_PENALIZER', 'pid_FUMBLER', 
    'pid_FUMBLE_RECOVERER', 'pid_FUMBLE_FORCER',
    'pid_INTERCEPTOR', 'pid_KICKER', 'pid_PUNTER', 
    'pid_RETURNER', 'pid_SACKER', 'pid_LATERALER',
    'pid_OTHER', 'pid_BLOCKER'
]

def get_custom_ents(sentence: str):
    doc = nlp(sentence)
    return [Entity(ent, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]

def get_all_row_ents(sentence: str):
    """
    Get all enities as dictionary
    Args:
        sentence (str): detail
    """
    if 'Timeout' in sentence: # no team names from timeouts - TOO MANY ERRORS
        return { e: np.nan for e in ALL_ENTS }
    doc = nlp(sentence)
    _dict = { e: [] for e in ALL_ENTS }
    for ent in doc.ents:
        if ent.label_ not in ALL_ENTS:
            print(sentence)
        _dict[ent.label_].append(f'{ent}:{ent.start_char}:{ent.end_char}')
    return { key: ('|'.join(_dict[key]) if len(_dict[key]) != 0 else np.nan) for key in _dict }

############################

# line = "Two Point Attempt: Jared Goff pass complete to to T.J. Hockenson for no gain"

# print(get_all_row_ents(line))

# ents = get_custom_ents(line)
# [e.show() for e in ents]
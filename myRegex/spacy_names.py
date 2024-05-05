from spacy import displacy
import spacy
from spacy.matcher import Matcher
from spacy.language import Language
import pandas as pd
import itertools
import regex as re
import numpy as np
import json

try:
    from namesRegex import getNames as getNames_re
except ModuleNotFoundError:
    from myRegex.namesRegex import getNames as getNames_re

patterns = [
    [{"POS": "PROPN"}, {"POS": "PROPN"}, {"ENT_TYPE": "PERSON"}],
    [{"POS": "PROPN"}, {"POS": "PROPN"}, {"POS": "PROPN"}, {"ENT_TYPE": "PERSON"}],
    [{"POS": "PROPN"}, {"POS": "PROPN"}],
    [
        {"POS": "PROPN", "OP": "+"},  # One or more proper nouns
        {"ORTH": "'", "OP": "?"},     # Optional apostrophe
        {"POS": "PROPN", "OP": "+"}   # One or more proper nouns
    ],
    [
        {"POS": "PROPN", "OP": "+"},   # One or more proper nouns
        {"TEXT": {"REGEX": "-"}, "OP": "?"},  # Optional hyphen
        {"POS": "PROPN", "OP": "+"}    # One or more proper nouns
    ],
    [
        {"POS": "PROPN"},            # Proper noun (first name)
        {"TEXT": {"REGEX": "'"}, "OP": "?"},  # Optional apostrophe
        {"POS": "PROPN"},            # Proper noun (last name)
        {"TEXT": {"REGEX": "-"}, "OP": "?"},  # Optional hyphen
        {"POS": "PROPN"},            # Proper noun (additional part of the last name, if any)
    ],
    [
        {"TEXT": {"REGEX": r"Tank"}},
        {"POS": "PROPN"}
    ],
    [
        {"TEXT": {"REGEX": r"Equanimeous"}},
        {"POS": "PROPN"},
        {"POS": "PROPN"}
    ],
    [
        {"TEXT": {"REGEX": r"Darnell"}},
        {"POS": "PROPN"}
    ],
    [
        {"TEXT": {"REGEX": r"Case"}},
        {"POS": "PROPN"}
    ],
    [
        {"POS": "PROPN"},
        {"POS": "PROPN"},
        {"TEXT": {"REGEX": r"III"}}
    ],
    [
        {"POS": "PROPN"},
        {"TEXT": {"REGEX": r"Manning"}}
    ],
    [
        {"POS": "PROPN"},
        {"TEXT": {"REGEX": r"Line"}}
    ]
]

nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)
matcher.add("ALL_NAMES", patterns)

PENALTY_TOKENS = list(pd.read_csv("%s.csv" % "D:/NFLPredictions3/myRegex/penaltyTokens")['name'].values)

###########################################

def convertPenalties(line):
    for token in PENALTY_TOKENS:
        line = line.replace((token + " "), '\n')
        line = line.replace((token + ","), '\n')
        if line.endswith(token):
            line = line.replace(token, '\n')
    return line

def replaceWords(sentence: str):
    # replace KC-9 or JAC-2 or KAN-19 or DAL--7 or DAL--17
    sentence = re.sub(r"[A-Z]{2,3}\-{1,2}[0-9]{1,2}", "", sentence)
    # -1 or -17
    sentence = re.sub(r"\-[0-9]+", "", sentence)
    # # TEN
    # sentence = re.sub(r"[A-Z]{3}", "", sentence)
    # 23
    sentence = re.sub(r"[0-9]+", "", sentence)
    # touchdown\s
    sentence = sentence.replace("touchdown ", "touchdown. ")
    # yards
    sentence = sentence.replace("yards ", "\n")
    sentence = re.sub(r"yards$", "", sentence)
    # yard
    sentence = sentence.replace(" yard ", "\n")
    sentence = re.sub(r" yard$", "", sentence)
    # :
    sentence = sentence.replace(":","")
    # kneels
    sentence = sentence.replace("kneels", "(dkskfsjdjfskg)")
    # \sand\s
    sentence = sentence.replace(" and ", "\n")
    # \spass\s
    sentence = sentence.replace(" pass", " (dkskfsjdjfskg) ")
    # ..
    sentence = sentence.replace("..", ".")
    return sentence

def getNames(sentence: str):
    if pd.isna(sentence):
        return np.nan
    sentence = replaceWords(sentence)
    sentence = convertPenalties(sentence)
    doc = nlp(sentence)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON" and len(ent.text.split(" ")) > 1]
    matches = matcher(doc)
    for _, start, end in matches:
        span = doc[start:end]  # The matched span
        if span.text not in names:
            names.append(span.text)
    names = list(set(names))
    correct_names = names.copy()
    for i in range(len(names)):
        for j in range(len(names)):
            n0, n1 = names[i], names[j]
            if n1 in n0 and len(n1) < len(n0) and '\n' not in n0:
                try:
                    correct_names.remove(n1)
                except ValueError:
                    continue
    # Ras-I in correct_names -> replace with Ras-I Dowling
    if 'Ras-I Dowling' in sentence and 'Ras-I' in correct_names:
        correct_names.remove('Ras-I')
        correct_names.append('Ras-I Dowling')
    # Ha Clinton-Dix in correct_names -> replace with Ha Ha Clinton-Dix
    if 'Ha Ha Clinton-Dix' in sentence and 'Ha Clinton-Dix' in correct_names:
        correct_names.remove('Ha Clinton-Dix')
        correct_names.append('Ha Ha Clinton-Dix')
    if 'Giovani Bernard middle' in sentence and 'Giovani Bernard middle' in correct_names:
        correct_names.remove('Giovani Bernard middle')
        correct_names.append('Giovani Bernard')
    correct_names = [n.replace(' for ','') for n in correct_names]
    return '|'.join([n.rstrip() for n in correct_names if ' yards ' not in n and '\n' not in n])

# ----------------------------------

def testDetails():
    file = json.load(open("test_details.json", "r"))
    count = 0
    for data in file:
        line, names = data['line'], data['names']
        names.sort()
        found_names = getNames(line).split("|")
        found_names.sort()
        if names != found_names:
            print(f"Names mismatch in '{line}'.")
            print(f"Found names: {found_names} <-> Expected names: {names}")
            count += 1
    if count == 0:
        print(f"All names correct.")
    return

################################

# testDetails()
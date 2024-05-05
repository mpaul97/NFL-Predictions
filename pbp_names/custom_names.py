import spacy
import os
import regex as re

nlp = spacy.load("D:/NFLPredictions3/pbp_names/models/names_model")

def get_names_custom(sentence: str):
    doc = nlp(sentence)
    return '|'.join(list(set([ent.text.rstrip() for ent in doc.ents if ent.label_ == "PERSON"])))

def get_name_indices(sentence: str, name: str):
    occs = re.finditer(f"{name}", sentence)
    return [(o.start(), o.end()) for o in occs]
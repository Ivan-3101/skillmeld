import pickle
from data import finance_prospects_dict

def load_model():
    return pickle.load(open("finance.pkl", "rb"))

def get_finance_prospects_dict():
    return finance_prospects_dict

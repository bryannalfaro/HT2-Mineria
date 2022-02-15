# Referencia forma > https://towardsdatascience.com/splitting-the-text-column-and-getting-unique-values-in-python-367a9548d085
from collections import Counter
import pandas as pd
def clean_data(list_data):
    list_splitted = list_data.str.cat(sep='|')
    list_final = list_splitted.split("|")
    list_split_empty = [x.strip("") for x in list_final]
    count = Counter(list_split_empty)
    df = pd.DataFrame(count.most_common(10000), columns=['Item', "Count"])
    return df

def clean_numeric_data(list_data, size_filter, size=200, keep_size=False):
    list_splitted = list_data.str.cat(sep='|')
    list_final = list_splitted.split("|")
    list_no_strings = [ try_float(x) for x in list_final ]
    list_no_nan = list_no_strings
    if not keep_size:
        list_no_nan = [x for x in list_no_strings if x is not None]
    if size_filter:
        if keep_size:
            list_no_nan = [ filter_size(x, size) for x in list_no_nan ]
        else:
            list_no_nan = [x for x in list_no_nan if x < size]
    count = Counter(list_no_nan)
    df = pd.DataFrame(count.most_common(10000), columns=['Item', "Count"])
    return df

def try_float(x):
    try:
        return float(x)
    except ValueError:
        return None

def filter_size(x, size):
    if x is not None:
        if x < size:
            return x
    return None

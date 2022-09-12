import pandas as pd

def prepare_relation_labels(data: pd.DataFrame) -> pd.DataFrame:
    ''' Changes all occurances of "agree", "disagree" and "discuss" to "related"'''
    exp = data["stance"] != "unrelated"
    data.loc[exp, "stance"] = "related"
    return data
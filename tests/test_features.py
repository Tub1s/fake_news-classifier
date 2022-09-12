import pandas as pd
from features.preprocessing import prepare_relation_labels

def test_relation_labels():
    test_df = pd.DataFrame({"stance": ["unrelated", "unrelated", "related", "related"]})
    assert prepare_relation_labels(pd.DataFrame({"stance": ["unrelated", "unrelated", "related", "related"]})).equals(test_df)
    assert prepare_relation_labels(pd.DataFrame({"stance": ["unrelated", "unrelated", "discuss", "agree"]})).equals(test_df)
    assert prepare_relation_labels(pd.DataFrame({"stance": ["unrelated", "unrelated", "disagree", "agree"]})).equals(test_df)
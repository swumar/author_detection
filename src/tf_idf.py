import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

df_train = pd.read_csv('../dataset/train.csv')
df_test = pd.read_csv('../dataset/test.csv')


print(len(df_train['text'].values.tolist()))
print(df_train.shape[0])
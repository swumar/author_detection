import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../dataset/train.csv')

df['text'] = df['text'].apply(lambda x: x.lower())

df.author = pd.Categorical(df.author).codes

df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)

vectorizer = TfidfVectorizer(stop_words='english')
fitter = vectorizer.fit(df_train['text'].values.tolist())

X_train = vectorizer.transform(df_train['text'].values.tolist())
y_train = df_train.author
X_test = vectorizer.transform(df_test['text'].values.tolist())
y_test = df_test.author

clf = LogisticRegression(random_state=0,max_iter=1000).fit(X_train, y_train)
y_pred = clf.predict(X_test)

class_names = ['EAP','HPL','MWS']
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels())
hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels())
plt.ylabel('True')
plt.xlabel('Predicted');
plt.show()

print(f1_score(y_test,y_pred,average=None))
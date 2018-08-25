#文本特征提取

#Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
#分词和统计单词出现频数
corpus=[
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X=vectorizer.fit_transform(corpus)
print(X.toarray())
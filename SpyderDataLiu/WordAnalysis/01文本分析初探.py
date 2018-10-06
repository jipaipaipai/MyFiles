# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 09:16:47 2018

@author: haoyu
"""

###读入文本
import pandas as pd
import jieba
import numpy

df_news=pd.read_table("./data/val.txt",names=['category','theme','URL','content'],encoding='utf-8')
df_news=df_news.dropna()
df_news.head()

content=df_news.content.values.tolist()
print(content[1000])

content_S=[]
for line in content:
    current_segment=jieba.lcut(line)
    if len(current_segment)>1 and current_segment !='\r\n':
        content_S.append(current_segment)
content[1000]

df_content=pd.DataFrame({'content_S:content_S})
df_content.head()

###数据清洗
stopwords=pd.read_csv('stopwords.txt',index_col=False,sep='\t',quoting=3,names=['stopword'],encoding='utf-8')
stopwords.head(20)

def drop_stopwords(contents,stopwords):
    contents_clean=[]
    all_words=[]
    for line in contents:
        line_clean=[]
        for word in line:
            if word in stopwprds:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean,all_words
    #print(contents_clean)

contents=df_content.content_S.value.tolist()
stopwords=stopwords.stopword.values.tolist()
contents_clean,all_words=drop_stopwords(contents,stopwords)

df_content=pd.DataFrame({'content_S':content_S})
df_content.head()

df_all_words=pd.DataFrame({'all_words':all_words})
df_all_words.head()

word_count=df_all_words.groupby(by=['all_words'])['all_words'].agg({'count':numpy.size})
word_count=word_count.reset_index(.sort_values(by=['count'],ascending=False)
word_count.head()

#词云
from wordcloud import WordColud
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.rcParams['figure.figsize']=(10.0,5.0)

wordcloud=WordCloud(font_path='./data/simhei.ttf',background_color='white',max_font_size=80)
word_frequence={x[0]:x[1] for x in words_count.head(100).values}
wordcloud=wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)

###TF-IDF
import jieba.analyse
index=1000
print(df_news['content'][index])
content_S_str=''.join(content_S[index])
print(' '.join(jieba.analyse.extract_tags(content_S_str,topK=5,withWeight=False)))

###LDA主题模型
from gensim import corpora,models,similarities
import gensim

dictionary=corpora.Dictionary(contents_clean)
corpus=[dictionary.doc2bow(sentence) for sentence in contents_clean]

lda=gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=20)#类似K-meansz,自己指定k值

print(lda.print_topic(1,topn=5))

###贝叶斯分类器





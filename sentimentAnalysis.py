#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.feature_extraction.text import CountVectorizer
count=CountVectorizer()


# In[2]:


data=pd.read_csv("Train.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


fig=plt.figure(figsize=(5,5))
colors=["skyblue",'pink']
pos=data[data['label']==1]
neg=data[data['label']==0]
ck=[pos['label'].count(),neg['label'].count()]
legpie=plt.pie(ck,labels=["Positive","Negative"],
                 autopct ='%1.1f%%', 
                 shadow = True,
                 colors = colors,
                 startangle = 45,
                 explode=(0, 0.1))


# In[6]:


df=["Hey Jude, refrain Dont carry the world upon your shoulders For well you know that its a fool Who plays it cool By making his world a little colder Na-na-na,a, na Na-na-na, na"]
bag=count.fit_transform(df)
print(count.get_feature_names())


# In[7]:


print(bag.toarray())


# In[8]:


import re
def preprocessor(text):
             text=re.sub('<[^>]*>','',text)
             emojis=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
             text=re.sub('[\W]+',' ',text.lower()) +                ' '.join(emojis).replace('-','')
             return text   


# In[9]:


preprocessor(data.loc[0,'text'][-50:])


# In[10]:


preprocessor("<a> this is :(  aweomee wohhhh :)")


# In[11]:


data['text']=data['text'].apply(preprocessor)


# In[12]:


from nltk.stem.porter import PorterStemmer

porter=PorterStemmer()


# In[13]:


def tokenizer(text):
        return text.split()


# In[14]:


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# In[15]:


tokenizer("Haters love Hating as they Hate")


# In[16]:


tokenizer_porter("Haters love Hating as they Hate")


# In[17]:


import nltk
nltk.download('stopwords')


# In[18]:


from nltk.corpus import stopwords
stop=stopwords.words('english')


# In[22]:


pip install wordcloud


# In[24]:


from wordcloud import WordCloud
positivedata = data[ data['label'] == 1]
positivedata =positivedata['text']
negdata = data[data['label'] == 0]
negdata= negdata['text']


# In[25]:


def wordcloud_draw(data, color = 'white'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                              if(word!='movie' and word!='film')
                            ])
    wordcloud = WordCloud(stopwords=stop,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(10, 7))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


# In[26]:


print("Positive words are as follows")
wordcloud_draw(positivedata,'white')


# In[27]:


print("Negative words are as follows")
wordcloud_draw(negdata)


# In[28]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None,tokenizer=tokenizer_porter,use_idf=True,norm='l2',smooth_idf=True)


# In[29]:


y=data.label.values
x=tfidf.fit_transform(data.text)


# In[30]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.5,shuffle=False)


# In[31]:


from sklearn.linear_model import LogisticRegressionCV

clf=LogisticRegressionCV(cv=6,scoring='accuracy',random_state=0,n_jobs=-1,verbose=3,max_iter=500).fit(X_train,y_train)

y_pred = clf.predict(X_test)


# In[32]:


from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:





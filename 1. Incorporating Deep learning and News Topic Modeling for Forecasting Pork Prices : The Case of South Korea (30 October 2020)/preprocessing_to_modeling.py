import pandas as pd
from konlpy.tag import Okt

okt = Okt()

# 데이터 전처리

# 문장을 받으면 동사와 명사만 추출해주는 함수
def NounOrVerb(sentence):
    lists=okt.pos(sentence)
    words_list=[]
    for word in lists:
        if word[1]=='Noun' or word[1]=='Verb':
            words_list.append(word[0])
    
    
    return words_list

# 저장된 기사 정보를 받고 결측치 제거 및 인덱스 초기화
articles=pd.read_csv('article_lists.csv')
articles=articles.dropna()
articles=articles.reset_index()
articles.drop(['index','Unnamed: 0'],axis=1,inplace=True)


# 뉴스 기사에서 명사와 동사만 추출
articles['words'] = articles['article'].apply(lambda x : NounOrVerb(x) if isinstance(x,str) else x)
articles.to_csv('article_pos.csv')


# tokenized 된 단어들을 이용하여 corpus 생성
df=pd.read_csv('article_pos.csv')
wordsss=list(df['words'])
words_corpus = []
for article in wordsss:
    if isinstance(article,list):
        s = " ".join(article)
        words_corpus.append(s)


# corpus를 이용하여 DTM 생성
from sklearn.feature_extraction.text import CountVectorizer

vector = CountVectorizer()

DTM = vector.fit_transform(words_corpus)

# DTM 생성 후 각 번호마다 무슨 단어인지 나타내주고 
# CSV파일로 저장
word_number=vector.vocabulary_
word_number=sorted(word_number.items(),key=lambda x:x[1])
DTM=pd.DataFrame(X,columns=word_number)
DTM.to_csv('DTM.csv')


from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=6)

lda.fit_transform(DTM)

# 키워드를 토픽별로 그룹화 시켜주는 함수
def make_topic_group(components, feature_names, n=10):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n - 1:-1]])
        print('\n')


terms = vector.get_feature_names()
make_topic_group(lda.components_,terms)


# 토픽 모델링 결과

# 문서 별 토픽 분포
doc_topics = lda.transform(DTM)
print(doc_topics.shape)
print(doc_topics[0])
print(doc_topics[1])


# 뉴스 기사의 토픽 별 단어 분포(1979,6) 와 단어 별 토픽 분포(6,13184)의 내적
# 기사 별 단어의 중요도 산출 (1979,13184)
importance_of_words=doc_topics.dot(lda.components_)

#tf-idf (1979,13184)를 생성하고 단어의 중요도와 원소 별 연산
from sklearn.feature_extraction.text import TfidfVectorizer

tf_idfvector = TfidfVectorizer()
tfidf=tf_idfvector.fit_transform(words_corpus).toarray()

LDA_tfidf=importance_of_words*tfidf
LDA_tfidf=pd.DataFrame(LDA_tfidf)

# 날짜 별 가격 정보와 벡터화된 기사 정보를 병합
articles_tmp=articles.reset_index()
LDA_tfidf_tmp=LDA_tfidf.reset_index()
dataset_before_FS=pd.merge(articles_tmp,LDA_tfidf_tmp)
dataset_before_FS=dataset_before_FS.drop(['index','article','words'],axis=1)

price_match={'2019.01':1723,'2019.02':1684,'2019.03':1690,'2019.04':1875,'2019.05':1977,'2019.06':1936,'2019.07':1931,'2019.08':1892,'2019.09':2056,'2019.1':1889,'2019.11':1680,'2019.12':1765,
 '2020.01':1690,'2020.02':1623,'2020.03':1887,'2020.04':1949,'2020.05':2273,'2020.06':2382,'2020.07':2324,'2020.08':2376,'2020.09':2345,'2020.1':2301,'2020.11':2131,'2020.12':2149,
 '2021.01':2113,'2021.02':2075,'2021.03':2043,'2021.04':2234,'2021.05':2451,'2021.06':2543,'2021.07':2599,'2021.08':2607,'2021.09':2635,'2021.1':2586,'2021.11':2522,'2021.12':2704,
 '2022.01':2361,'2022.02':2352,'2022.03':2344,'2022.04':2353}
dataset_before_FS['price']=dataset_before_FS['date'].apply(lambda x: price_match[str(x)])


#데이터 월별 집계
dataset_before_FS=dataset_before_FS.groupby(['date','price']).sum()

#병합 후 인덱스 정리
dataset_before_FS.to_csv('dataset_before_FS.csv')
dataset_before_FS=pd.read_csv('dataset_before_FS.csv')

#pearson 상관계수 0.4 이상인 값들만 추출 (129개)
data_corr=dataset_before_FS.corr(method='pearson')['price']
fs_feature=np.where(abs(data_corr)>0.4)

fs_columns=[]
for i in list(fs_feature[0]):
    fs_columns.append(str(i))

fs_columns.append('price')
dataset=dataset_before_FS[fs_columns]


# 학습 데이터셋과 테스트 데이터셋 분리 
# (35개월 데이터로 최근 4달간 예측)
# train_x, train_y, test_x, test_y
train = dataset.iloc[:-4]
test = dataset.iloc[-4:]
train_x = np.array(train.loc[:])
train_y= np.array(train['price'])
test_x = np.array(test.loc[:])
test_y= np.array(test['price'])

# (1, 3, 130)에 맞게 구성
def make_dataset(train_x):
    trainX = []
    cnt=len(train_x)
    for i in range(0,cnt):
        if cnt==i+2:
            break
        tmp = np.column_stack([train_x[i],train_x[i+1],train_x[i+2]])
        trainX.append(tmp.T)
    
    return np.array(trainX)

# 학습용 데이터 trainX와 trainY 완성
trainX = make_dataset(train_x)

trainY=[]
for i in train_y[2:]:
    trainY.append(i)
trainY= np.array(trainY)

from keras.models import Sequential
from keras.layers import Dense, LSTM
 
model = Sequential()
model.add(LSTM(5,batch_input_shape=(1,3,130)))
model.add(Dense(1))

model.summary()
model.compile(optimizer='adam',loss='mse')
history= model.fit(trainX,trainY,epochs=10000,batch_size=1)

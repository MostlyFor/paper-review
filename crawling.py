import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import requests
import time

# 기사의 정보를 저장하는 dataframe
news_Data=pd.DataFrame(columns={'date','article'})

# url 주면 페이지 안에서 기사 내용들을 dataframe에 추가 저장해주는 함수
def url_crawling(url,df):
    res = requests.get(url)
    soup = BeautifulSoup(res.content, 'html.parser')
    #뉴스 기사 태그들
    title_list = soup.find_all('div', class_='list-titles table-cell')
    
    #뉴스 기사 가져와서 데이터 저장하기
    for title in title_list:
        k=title.find('a')
        link = 'http://www.pigtimes.co.kr/'+k.attrs['href']
        news=requests.get(link)
        news_ = BeautifulSoup(news.content, 'html.parser')

        #뉴스에서 월까지 날짜 가져오기
        date_updated=news_.find('span',class_='updated')
        date_updated_=date_updated.get_text()
        date_updated_month = date_updated_[:7]

        #뉴스에서 내용 가져오기 
        #article은 뉴스 기사 내용
        news_article=news_.find('div',id='article-view-content-div')
        article =''
        for content in news_article.select('p'):
            article += content.get_text()

        #데이터 프레임에 추가
        df=df.append({'date':date_updated_month,'article':article},ignore_index=True)
        
    return df

# 각 페이지 기사 링크를 관찰해본 결과 나머지 주소는 똑같고 page_n만 바꾸면 됨.
for page_n in tqdm(range(1,100)):
    
    url='http://www.pigtimes.co.kr/news/articleList.html?page='+str(page_n)+'&total=21005&sc_section_code=S1N7&sc_sub_section_code=&sc_serial_code=&sc_area=&sc_level=&sc_article_type=&sc_view_level=&sc_sdate=&sc_edate=&sc_serial_number=&sc_word=&sc_word2=&sc_andor=&sc_order_by=E&view_type='
    news_Data=url_crawling(url,news_Data)

news_Data.to_csv('article_lists.csv')



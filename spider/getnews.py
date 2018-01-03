#coding:utf-8

from bs4 import BeautifulSoup
import requests
import os
import sys
import codecs


news_url = 'http://news.sina.com.cn/china/'
web_data = requests.get(news_url)
# 使用UTF-8编码
web_data.encoding = 'utf-8'

# 使用剖析器为html.parser
soup = BeautifulSoup(web_data.text,'html.parser')

#新建文件
path = r'E:\txt'
if os.path.exists(path) == False:  
    os.mkdir(path);

f = codecs.open(path+r'\new5.txt','w','utf-8')

i = 0

#遍历每一个class=news-item的节点
for news in soup.select('.news-item'):
    h2 = news.select('h2')
    #只选择长度大于0的结果
    if len(h2)>0:
        #新闻时间
        time = news.select('.time')[0].text
        #新闻标题
        title = h2[0].text
        #新闻链接
        href = h2[0].select('a')[0]['href']

        s = str(i)+r'.'+title+':'+href+'\r\n'
        #打印
        #print str

        #保存
        f.write(s)

        i+=1
        if i>10:
            break


f.close()
    

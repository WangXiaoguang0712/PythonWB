#coding:utf-8

import urllib
import urllib2
from bs4 import BeautifulSoup
import chardet
import pdb

#读取页面文本流
def getHtml(url,data):
    request = urllib2.Request(url,data)
    html = urllib2.urlopen(request).read().decode('utf-8')
    return html

#发送请求到指定地址
def getPageInfo(index):
    data = {
            'CategoryId':808,
            'CategoryType':"SiteHome",
            'ItemListActionName':"PostList",
            'PageIndex:index':index,
            'ParentCategoryId':0,
            'TotalPostCount':4000
            }
    url = 'https://www.cnblogs.com/mvc/aggsite/postlist.aspx'
    data = urllib.urlencode(data)
    return getHtml(url,data)


#获取blog属性
def getDicItem(bd):
    d_item = {}
    #获取标题
    t = bd.find_all('a',attrs={'class':'titlelnk'})
    blog_title = t[0].string.strip( )
    blog_lnk = t[0]['href']
    d_item['blog_title']=blog_title
    d_item['blog_lnk']=blog_lnk
    
    #获取作者
    t = bd.find_all('a',attrs={'class':'lightblue'})
    blog_author = t[0].string
    d_item['blog_author']=blog_author
    #获取时间
    t = bd.find_all('div',attrs={'class':'post_item_foot'}) 
    t = t[0].a.next_sibling.replace('\r\n','').replace('\t','')
    #t = t.decode('unicode').encode('utf-8')
    rp_str = '发布于'.decode('utf-8')
    blog_ctime = t.replace(rp_str,'').lstrip(' ').rstrip(' ')
    d_item['blog_ctime']=blog_ctime

    #获取文章内容
    blog_html = getHtml(blog_lnk,'')
    blog_soup = BeautifulSoup(blog_html,'html.parser')
    blog_text = blog_soup.find('div',attrs={'id':'cnblogs_post_body'})
    d_item['blog_text']=blog_text
    #print(tag_blogtext.text)
    return d_item

#获取blog列表
def getList(n1):
    l_item = []
    for n in range(1,n1+1):
        print('开始读取第'+str(n)+'页')
        html_data = getPageInfo(n)
        soup = BeautifulSoup(html_data,'html.parser')
        soup_bd = soup.find_all('div',attrs={'class':'post_item_body'},limit=10)
        for bd in soup_bd:
            item = getDicItem(bd)
            l_item.append(item)
        print('结束读取第'+str(n)+'页')
    return l_item



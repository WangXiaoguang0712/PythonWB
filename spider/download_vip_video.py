# _*_ coding:utf-8 _*_
import urllib
from bs4 import BeautifulSoup
from urlparse import urlparse
import os,chardet,re
__author__ = 'T'

def load_detail(url):
    page = urllib.urlopen(url)
    soup = BeautifulSoup(page, 'html.parser')
    #uri = soup.find(attrs={"id":"Zoom"}).find('table')[0].find('a')
    uri = soup.find(attrs={"id":"Zoom"}).find('table').find('a')
    print uri['href']
    # 调用迅雷下载
    os.execl(r'C:\Program Files (x86)\Thunder\Program\Thunder.exe ','-StartType:DesktopIcon',uri['href'])


def dowload_ygdy(page):
    url = 'http://www.ygdy8.com/html/gndy/dyzz/list_23_{0}.html'.format(page)
    parse_res = urlparse(url)
    host = parse_res[0] + r'://' + parse_res[1]
    page = urllib.urlopen(url).read().decode('GB2312', 'ignore')

    soup = BeautifulSoup(page, 'html.parser')
    video_div = soup.find(attrs={"class":"co_content8"})
    video_tab = video_div.findAll(attrs={"class":"tbspan"})
    for v in video_tab:
        v_m = v.find(attrs={"class":"ulink"})
        v_title = v_m.string
        if re.search(u'联盟|营救',v_title):
            # print v_m.get('href')
            v_url = v_m['href']
            v_date = v.find('font').string
            print(v_title.encode('gbk', 'ignore'))
            #load_detail(host + v_url)

def start_dowload():
    for i in range(1,11):
        print('start parse page {0}'.format(i))
        dowload_ygdy(i)

if __name__ == "__main__":
    start_dowload()
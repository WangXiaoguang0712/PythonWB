#-*- coding:utf-8 -*-

import blog_parser
import os
import MySQLdb as myd
import codecs
import chardet
import json

# 文件存储
def saveToText(l_item):
    file_path=r'E:\txt\blog'
    if os.path.exists(file_path) == False:
        os.mkdir(file_path)
    
        
    f = codecs.open(file_path+r'\blog.txt','w','utf-8')
    for item in l_item:
        f.write(json.dumps(item,encoding='utf-8',ensure_ascii=False))
        f.write('\r\n')
    f.close()

# mysql 存储
def execNoQuery(sql):
    conn = myd.connect(
        host='192.168.48.131',
        port=3306,
        user='mysql',
        passwd='mysql',
        db='test',
        charset= 'utf8'
        )
    cur = conn.cursor()
    cur.execute(sql)
    cur.close()
    conn.commit()
    conn.close()
    
    
# begin
#pdb.set_trace() 
#_item = blog_parser.getList(2)
#saveToText(l_item)

l_item = blog_parser.getList(1)
for item in l_item:
    str1 = item['blog_title'].encode('utf-8')
    btle = myd.escape_string(str1)
    blnk = myd.escape_string(item['blog_lnk'].encode('utf-8'))
    baut = myd.escape_string(item['blog_author'].encode('utf-8'))
    bctm = myd.escape_string(item['blog_ctime'].encode('utf-8'))
    btxt = myd.escape_string(item['blog_text'].encode('utf-8'))
    #sql = "insert into Py_Blogs values('s%','s%','s%','s%',null)" %(btle,blnk,baut,bctm)
    sql = "insert into Py_Blogs values('"+btle+"','"+blnk+"','"+baut+"','"+bctm+"','"+btxt+"')"
    
    
    #print sql
    execNoQuery(sql)



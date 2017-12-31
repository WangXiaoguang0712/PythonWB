import urllib
import sys
import sqlite3
import os

__author__ = 'T'

def schedual(a, b, c):
    per = 100.0 * a * b / c
    if per > 100 :
        per = 100
    #print '%.2f%%' % per
    sys.stdout.write("\r%.2f%%" % per + ' complete')
    sys.stdout.flush()

def downloadnow():
    weburl = ''
    local_path = r''
    urllib.urlretrieve(weburl,local_path,schedual)

downloadnow()

class DB(object):
    def __init__(self,dbname='test'):
        self.conn = sqlite3.connect(dbname)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>> db %s connected!>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>' %(dbname))

    def execute_sql(self,sql):
        if len(sql) == 0:
            raise ValueError('please input correct sql')
        else:
            cur = self.conn.cursor()
            records = cur.execute(sql)
            print cur.rowcount
            for line in records:
                print line
            cur.close()
#db = DB('test1')
#db.execute_sql('select * from sqlite_master')
#db.execute_sql('create table first_table (id int PRIMARY KEY ,NAME VARCHAR(30),memo VARCHAR(200))')
#db.execute_sql("insert into first_table values(1,'twi','he is bad guy')")


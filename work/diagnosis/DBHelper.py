# _*_ coding:utf-8 _*_
import types
import ConfigParser
import os
import sys
import time
from util import segment
import pymysql
import pymssql
__author__ = 'T'
root_path = sys.path[0] + os.sep

class Log(object):
    def __init__(self):
        log_path = os.path.dirname(sys.path[0])
        filename = time.strftime('%Y%m%d',time.localtime()) + '.log'
        self.full_name = os.path.join(log_path, 'log', filename)

    def logging(self, msg):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        with open(self.full_name, 'a') as f:
            f.write(timestamp + '\t' + msg.encode('utf8') + '\r\n')


class DBHelper():
    def __init__(self, db_type='mysql'):
        # 加载配置文件
        self.config = ConfigParser.ConfigParser()
        self.config.read(root_path + 'Config.properties')
        self.connstr = ""
        self.db_type = db_type
        if db_type == "mysql":
            #_/A:v.u8*ky-
            self.connstr = { 'host': self.config.get('dbconf', 'db.host'),
                             'port': int(self.config.get('dbconf', 'db.port')),
                             'user': self.config.get('dbconf', 'db.username'),
                             'passwd': self.config.get('dbconf', 'db.password'),
                             'db': self.config.get('dbconf', 'db.name'),
                             'charset': self.config.get('dbconf', 'db.charset')}
        else:
            self.connstr = { 'host': '127.0.0.1', 'port': 1433, 'user': 'sa', 'passwd':'haosql',
                             'db': 'textsearch','charset': 'utf8'}

    def select(self, sql):
        conn = None
        if self.db_type == "mysql":
            conn = pymysql.connect(host= self.connstr['host'], port=self.connstr['port'], user=self.connstr['user'],
                                   passwd=self.connstr['passwd'], db=self.connstr['db'] ,charset=self.connstr['charset'])
        else:
            conn = pymssql.connect(host= self.connstr['host'], port=self.connstr['port'], user=self.connstr['user'],
                                   password=self.connstr['passwd'], database=self.connstr['db'] ,charset=self.connstr['charset'])
        cursor = conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchall()
        cursor.close()
        conn.close()
        return res

    def execsql(self, sql):
        conn = None
        if self.db_type == "mysql":
            conn = pymysql.connect(host= self.connstr['host'], port=self.connstr['port'], user=self.connstr['user'],
                                   passwd=self.connstr['passwd'], db=self.connstr['db'] ,charset=self.connstr['charset'])
        else:
            conn = pymssql.connect(host= self.connstr['host'], port=self.connstr['port'], user=self.connstr['user'],
                                   password=self.connstr['passwd'], database=self.connstr['db'] ,charset=self.connstr['charset'])
        cursor = conn.cursor()
        cursor.execute(sql)
        cursor.close()
        conn.commit()
        conn.close()

    def getallsymptom(self, icd):
        """根据icd取得该疾病的典型症状"""
        symplist = []
        #连接数据库
        conn = None
        if self.db_type == "mysql":
            conn = pymysql.connect(host= self.connstr['host'], port=self.connstr['port'], user=self.connstr['user'],
                                   passwd=self.connstr['passwd'], db=self.connstr['db'] ,charset=self.connstr['charset'])
        else:
            conn = pymssql.connect(host= self.connstr['host'], port=self.connstr['port'], user=self.connstr['user'],
                                   password=self.connstr['passwd'], database=self.connstr['db'] ,charset=self.connstr['charset'])
        cur = conn.cursor()
        sql = 'SELECT label,classsymp,symps FROM symptab where label = %s'
        param = (icd)
        cur.execute(sql,param)
        #取得所有的症状内容
        for row in cur:
            if row[2] != "" or row[2] is not None:
                symps = row[2].decode("utf-8").split(',')
                classsymp = row[1].decode("utf-8")
                symps.insert(0, classsymp)
                symplist.append(symps)
            else:
                classsymp = row[1].decode("utf-8")
                symplist.append(classsymp)
        cur.close()
        conn.close()

        return symplist

    def geticdsymplist(self, content, icd):
        classsymresult = []
        #分词
        emrlist = segment(content.decode('utf-8'))
        #根据icd取得所有的症状
        symlist = self.getallsymptom(icd)
        for symps in symlist:
            map = {}

            updata = 0
            classresult = ''
            #判断典型症状的同义词是否在分词后的词表里
            if type(symps) is types.ListType:
                classsym = symps[0]
            else:
                classsym = symps
            if(classsym in  emrlist):
                classresult = classsym
                map['name'] = classresult
                map['value'] = 1
                updata = 1
            elif type(symps) is types.ListType:
                syms = symps[1:]
                for sym in syms:
                    if(sym in emrlist):
                        classresult = classsym + '(' + sym +')'
                        map['name'] = classresult
                        map['value'] = 1
                        updata = 1
                        break
                #某个典型症状循环结束，并且没有找到该词，则设定value为0
                if updata != 1 :
                    map['name'] = classsym
                    map['value'] = 0
            #只有1个症状的，并且不在病历中
            else:
                map['name'] = classsym
                map['value'] = 0
            classsymresult.append(map)
        return classsymresult

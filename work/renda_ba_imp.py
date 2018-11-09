# _*_ coding:utf-8 _*_

import pymysql
import chardet
import codecs

__author__ = 'T'

def imp_data(file_name, tab_name, n_commit):

    # open file

    timer = 0
    with open(file_name, 'r') as f:
        sql = "insert into " + tab_name + " values"
        lines = f.readlines()
        print(len(lines))
        for row in lines:
            timer += 1
            sql_values = "("
            for field in row.split('\t'):
                sql_values += "null," if field == '\N' or field == 'N' else "'" + field.replace('\'','').rstrip('\r\n') + "',"
            sql_values = sql_values.rstrip(',') + "),"
            sql += sql_values

            if timer % n_commit == 0 or timer == len(lines):
                sql = sql.rstrip(',')
                # print(sql.decode('utf8').encode('gbk'))
                # db connect
                conn = pymysql.connect(host="192.168.0.251", user='root', passwd="123456", db='renda', charset='utf8')
                c = conn.cursor()
                try:
                    c.execute(sql.decode('utf8'))
                except:
                    print(sql.decode('utf8').encode('gbk'))
                    print('ss')
                c.close()
                conn.commit()
                conn.close()
                sql = "insert into " + tab_name + " values"
                print(timer)


if __name__ == "__main__":
    # imp_data(u'D:\\临时文件\\rd_mb_ss.csv','bh_tjplat_wt4_22_surgery.csv', 10000)
    # imp_data(u'D:\\临时文件\\rd_ss.csv','bh_tjplat_wt4_22_surgery', 10000)
    """
    imp_data(u'D:\\临时文件\\rd_3.csv','bh_tjplat_wt4_22', 10000)
    imp_data(u'D:\\临时文件\\rd_4.csv','bh_tjplat_wt4_22', 10000)
    imp_data(u'D:\\临时文件\\rd_5.csv','bh_tjplat_wt4_22', 10000)
    imp_data(u'D:\\临时文件\\rd_remain.csv','bh_tjplat_wt4_22', 10000)
    imp_data(u'D:\\临时文件\\rd_mb.csv','bh_tjplat_wt4_22', 10000)
    """
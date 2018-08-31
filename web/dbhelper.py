# _*_ coding:utf-8 _*_
__author__ = 'T'

import os
import sqlite3

class DBHelper(object):
    def __init__(self, db_path):
        self.db_path = db_path

    def exec(self, sql, *args):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(sql, *args)
        conn.commit()
        c.close()
        conn.close()

    def select(self, sql):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(sql)
        res = c.fetchall()
        c.close()
        conn.close()
        return res

if __name__ == "__main__":
    pass
# _*_ coding:utf-8 _*_
__author__ = 'T'

import os, sys,  datetime, math, re,  jieba.analyse
from array import array
from DBHelper import DBHelper
import numpy as np
from collections import defaultdict
import ConfigParser
root_path=sys.path[0] + os.sep
# import  redis


class TextSearch():

    #成员变量
    keywordList = {}    #关键词列表，为字典，key为主题名、value为另一个字典；value字典中的key为关键词字数、value为同长度的关键词的列表。
                        #示例：
                        #keywordList = {
                        #    'zhengzhuang': {
                        #        2: ['word1', 'word2', 'word3'],
                        #        3: ['word4', 'word5', 'word6']
                        #    }
                        #}
    synonymList = [] #近义词列表
    prePosWords = u'很,非常,特别,严重,连续,多日,持续,明显,反复'.split(',') #前置加强词
    afterPosWords = u'明显,多日,加重,剧烈'.split(',') #后置加强词
    preNegWords = u'无,不,未,未见,未现,否认,不伴,未闻及'.split(',') #前置否定词
    afterNegWords = u'不多,不大,不明显,阴性'.split(',') #后置否定词
    symbolList = array('u', u'，。：；“”（）‘’ ') #用于分隔语句的标点符号

    #初始化
    def __init__(self):
        # 加载配置文件
        self.config = ConfigParser.ConfigParser()
        self.config.read(root_path + 'Config.properties')
        #加载关键词列表
        db = DBHelper()
        # self.c = redis.Redis(host='127.0.0.1', port=6379, db=0)
        # rows_keyword = eval(self.c.get('keyword'))
        self.rows_keyword =  db.select('select Keyword, SubjectCode,map from DOC_Keyword')
        for row in self.rows_keyword:
            keyword = row[0]
            subject = row[1]
            klen = len(keyword)
            if not subject in self.keywordList:
                self.keywordList[subject] = {}
            if not klen in self.keywordList[subject]:
                self.keywordList[subject][klen] = set()
            if not keyword in self.keywordList[subject][klen]:
                self.keywordList[subject][klen].add(keyword)

        # 加载惩罚系数及icd 与名称对应
        self.ill_penalize = {}
        self.ill_icd = {}
        db = DBHelper()
        rows_ill_penalize =  db.select('select icd_code,penalize_coef,icd_name from DOC_Ill')
        for row in rows_ill_penalize:
            self.ill_penalize[row[0]] = float(row[1])
            self.ill_icd[row[2]] = row[0]

        # 加载idf
        self.kwi_idf = {}
        db = DBHelper()
        rows_keyword_idf =  db.select('select Keyword, idf from DOC_Keyword_idf')
        # rows_keyword_idf = eval(self.c.get('keyword_idf'))
        for kwi in rows_keyword_idf:
            self.kwi_idf[kwi[0]] = float(kwi[1])

        #加载近义词列表
        synonymFile = os.path.join(sys.path[0], self.config.get('modelconf', 'synonymword'))
        f = open(synonymFile, 'rb')
        lineIndex = 0
        for line in f.readlines():
            line_utf8 = line.decode('utf-8-sig').strip('\r\n').rstrip(' ')
            swset = set()
            for sw in line_utf8.split(' '):
                if sw != '':
                    swset.add(sw)
            self.synonymList.append(swset)
            lineIndex = lineIndex + 1
        f.close()

    #从字符串中取得其中一部分
    def seekString(self, source, pos, delta):
        l = len(source)
        if pos >= 0 and pos < l:
            if delta > 0:
                if pos + delta - 1 < l:
                    return source[pos : pos+delta]
            else:
                if delta < 0:
                    if pos + delta + 1 >= 0:
                        return source[pos + delta + 1 : pos + 1]
        return ''

    #处理文本
    def procText(self, source, subject = ''):
        result = []
        maxKeywordLen = 0
        currentKeywordList = {}

        if len(source) < 2:
            return result

        t1 = datetime.datetime.now()

        if (subject in self.keywordList):
            #使用专业词表
            maxKeywordLen = max(self.keywordList[subject].keys())
            currentKeywordList = self.keywordList[subject]

            source = source + u'。'
            sentence_array = array('u')

            for ch in source:
                if not ch in self.symbolList and len(sentence_array) < 100:
                    sentence_array.append(ch)
                    continue

                #找出一句话
                slen = len(sentence_array)
                if slen > 0:
                    sentence = sentence_array.tounicode()

                    #找出一句话中的关键字
                    j = 0
                    while j < slen:
                    # for j in range(0, slen):
                        match = ''
                        mlen = min(slen, maxKeywordLen)
                        for keywordLen in range(mlen, 1, -1):

                            if (j + keywordLen <= slen):
                                word = self.seekString(sentence, j, keywordLen)

                                #print(str(j) + ',' + str(keywordLen) + ':' + word)
                                if len(word) > 0:
                                    if keywordLen in currentKeywordList:
                                        #print(keywordLen)
                                        if word in currentKeywordList[keywordLen]:

                                            match = word
                                            break
                        if len(match) == 0:
                            j += 1
                            continue
                        #print('Match:' + match)

                        #找出匹配关键词的加强词、否定词，评定得分
                        rating = 0
                        preWord = ''
                        afterWord = self.seekString(sentence, j + len(match), 2)
                        if len(afterWord) > 0:
                            if afterWord in self.afterPosWords:
                                rating = 2
                            elif afterWord in self.afterNegWords:
                                rating = -2
                            if rating == 0:
                                afterWord = '' #清除afterWord以防影响位移
                        if rating == 0:
                            preWord = self.seekString(sentence, j - 1, -3)
                            if len(preWord) > 0:
                                if preWord in self.prePosWords:
                                    rating = 2
                                elif preWord in self.preNegWords:
                                    rating = -2
                            if rating == 0:
                                preWord = self.seekString(sentence, j - 1, -2)
                                if len(preWord) > 0:
                                    if preWord in self.prePosWords:
                                        rating = 2
                                    elif preWord in self.preNegWords:
                                        rating = -2
                                if rating == 0:
                                    preWord = self.seekString(sentence, j - 1, -1)
                                    if len(preWord) > 0:
                                        if preWord in self.prePosWords:
                                            rating = 2
                                        elif preWord in self.preNegWords:
                                            rating = -2
                        if rating == 0:
                            rating = 1

                        #放入结果列表
                        newMatch = True
                        for m in result:
                            if m['keyword'] == match:
                                m['rating'] = m['rating'] + rating
                                newMatch = False
                                break
                        if newMatch:
                            result.append({ 'keyword': match, 'rating': rating, 'p': 0 })

                        #位移
                        j += len(match) + len(afterWord)

                    sentence_array = array('u')

        else:
            #使用jieba进行分析
            tags = jieba.analyse.extract_tags(source, 50, withWeight=True)
            #tags = jieba.analyse.textrank(source, 50, withWeight=True)
            for t in tags:
                result.append({ 'keyword': t[0], 'rating': t[1], 'p': 0 })

        #最后计算所有的P
        total_p = 0.0
        total_m = 0.0
        for m in result:
            if m['rating'] > 0:
                total_p += m['rating']
            elif m['rating'] < 0:
                total_m += m['rating']
        for m in result:
            _idf = 1
            if m['keyword'] in self.kwi_idf.keys():
                _idf = self.kwi_idf[m['keyword']]
            if m['rating'] > 0:
                m['p'] = m['rating'] / total_p * _idf
            elif m['rating'] < 0:
                m['p'] = m['rating'] / (-total_m) * _idf

        t2 = datetime.datetime.now()

        #print(result)
        dt = t2 - t1
        mdt = dt.seconds * 1000 + dt.microseconds / 1000
        #print('用时：' + str(dt))

        return result

    #搜索
    def search(self, condition, subject='', is_penalize=False):
        result = []
        ratingTable = 'DOC_Rating'
        if subject in self.keywordList:
            ratingTable = ratingTable + '_' + subject
        t0 = datetime.datetime.now()
        # queryWords = sorted(self.procText(condition, subject), key=lambda k:k['keyword'])
        queryWords = sorted(self.procText(condition, 'zhengzhuang'), key=lambda k:k['keyword'])
        t1 = datetime.datetime.now()
        if len(queryWords) == 1:
            queryWords.append({ 'keyword': '', 'rating': 1, 'p': 1 })
        # print('queryWords:' + str(len(queryWords)))
        keywordIndex = 0
        keywordCnt = {}
        sqlSelect = ''
        sql_subquery = "select docid"
        sql_p_pos = " 1=0 "
        sql_p_neg = " 1=0 "

        for item in queryWords:
            keywordIndex = keywordIndex + 1
            keyword = item['keyword']
            synonymWords = []
            keywordCnt[keyword] = keywordIndex

            #查找近义词
            for sl in self.synonymList:
                if keyword in sl:
                    #print(u'找到' + keyword + u'的近义词：' + str(sl))
                    for sword in sl:
                        if keyword != sword:
                            synonymWords.append(sword)
                    break

            #拼装join语句
            sqlSelect += ",v.p_" + str(keywordIndex)
            #拼接同义词
            sql_subquery += ",sum(case when Keyword='" + item['keyword'].encode('utf8') + "'"
            sql_syn = ""
            # print(synonymWords)
            for word in synonymWords:
                sql_syn += " or Keyword='" + word.encode('utf8')  + "'"
            #if keywordIndex > 1:
            #    sql_p_pos += " or"
            if item["rating"] > 0:
                sql_p_pos += " or Keyword='" + item['keyword'].encode('utf8')  + "'"
                sql_p_pos += sql_syn
            else:
                sql_p_neg += " or Keyword='" + item['keyword'].encode('utf8')  + "'"
                sql_p_neg += sql_syn
            sql_subquery += sql_syn
            if item['keyword'] == '':
                sql_subquery += " then 1 else 1 end)p_" + str(keywordIndex)
            else:
                sql_subquery += " then p else 0 end)p_" + str(keywordIndex)

        sql_subquery += " from " + ratingTable
        if len(sql_p_pos) > 0:
            sql_subquery += " where (p>0 and (" + sql_p_pos + "))"
        if len(sql_p_neg) > 0:
            sql_subquery += " or (p<0 and (" + sql_p_neg + "))"
        sql_subquery += " GROUP BY docid"
        #最后的SQL
        sql = "select c.PATNO, c.DocName,rtrim(c.icd)icd,rtrim(c.ill)ill " + sqlSelect + " from DOC_Content c "
        sql += "inner join (" + sql_subquery + ")v on c.docid=v.docid"
        # print (sql.decode('utf8'))
        # 到数据库执行查询
        db = DBHelper()
        rows =  db.select(sql)
        t2 = datetime.datetime.now()
        #计算相关度，生成结果列表

        for row in rows:
            v1 = 0
            v2 = 0
            v3 = 0
            """
            vec_kw = np.array([x['p'] for x in queryWords])
            vec_rp = np.array([row[3 + keywordCnt[x['keyword']]] for x in queryWords])
            score = vec_kw.dot(vec_rp.T) / ( np.sqrt(np.sum(vec_kw ** 2)) * np.sqrt(np.sum(vec_rp ** 2)))  # cos
            """
            for qw in queryWords:
                kp = float(row[3 + keywordCnt[qw['keyword']]])
                if kp is None:
                    kp = 0
                v1 = v1 + qw['p'] * kp
                v2 = v2 + qw['p'] * qw['p']
                v3 = v3 + kp * kp
            if v2 == 0 or v3 == 0:
                continue
            score = v1 / (math.sqrt(v2) * math.sqrt(v3))

            if is_penalize:
                score = score * self.ill_penalize[row[2]]
            # res = eval(self.c.hmget('content',row[0])[0])
            result.append({ 'PATNO': row[0], 'DocName': row[1], 'ICD': row[2], 'ill' : row[3], 'R': score })

        #排序
        result = sorted(result, key=lambda k:k['R'], reverse=True)

        t3 = datetime.datetime.now()
        print 'segment:', t1 - t0
        print 'query:', t2 - t1
        print 'calculate:', t3 - t2
        print 'total:', t3 - t0
        # print('得到：' + str(len(result)) + ' 项结果，查询用时：' + str(t2 - t1) + ', 计算用时：' + str(t3 - t2))

        #返回
        tt = t3 - t1

        return { 'data': result, 'milliseconds': tt.seconds * 1000 + tt.microseconds / 1000 }

    #高亮关键字
    def highlightText(self, source, condition, subject):
        #找出查询词，包括其同义词
        wordList = []
        queryWords = sorted(self.procText(condition, subject), key=lambda k:k['keyword'])

        for qw in queryWords:
            wordList.append(qw['keyword'])
            #查找近义词
            for sl in self.synonymList:
                if qw['keyword'] in sl:
                    #print(u'找到' + keyword + u'的近义词：' + str(sl))
                    for sword in sl:
                        if qw['keyword'] != sword:
                            wordList.append(sword)
                    break

        #生成正则条件
        pattern = ''
        for w in wordList:
            if len(pattern) > 0:
                pattern = pattern + '|'
            pattern = pattern + w

        #高亮处理
        def func_replace(m):
            return '<b>' + m.group() + '</b>'
        p = re.compile(pattern)
        source = p.sub(func_replace, source)

        #找到第一处命中关键词的位置
        m = p.search(source)
        if m is None:
            return source[0:300]

        n1 = m.start()
        if n1 > 10:
            n1 = n1 - 10
        if n1 + 300 < len(source):
            return source[n1:n1 + 300]
        else:
            return source[n1:len(source)]

    def diagnosis(self, condition, subject='', method='wv', is_penalize=False, is_return_emrs=False):
        db = DBHelper()
        ratingTable = 'DOC_Rating'
        sql_inner = ""
        result = {}
        icds = []
        emrs = {}
        if subject in self.keywordList:
            ratingTable = ratingTable + '_' + subject
        queryWords = sorted(self.procText(condition, 'zhengzhuang'), key=lambda k:k['keyword'])
        if len(queryWords) == 0:
            result = {"ICDS" : '', "rulers" : ''}
        else:
            # 求 典型症状
            symptom = []
            for qw in queryWords:
                for kw in self.rows_keyword:
                    if qw['keyword'] == kw[0] and kw[2] not in symptom:
                        symptom.append(kw[2])
            if method == "vector":
                l_words = []
                l_p = []
                for kw in queryWords:
                    sql_inner += "'"+ kw['keyword'].encode('utf-8') +"',"
                    l_words.append(kw['keyword'])  # 症状
                    l_p.append(kw['p'])  #  p 值
                sql = "select * from DOC_Rating_P where keyword in("+ sql_inner.rstrip(',') +") "
                r = db.select(sql)
                l_ill = list(set([x[1] for x in r]))  # 疾病
                m_data = np.zeros((len(l_ill), len(l_words)))
                for row in r:
                    m_data[l_ill.index(row[1])][l_words.index(row[0])] = row[2]
                v_idf = np.array(l_p) * len(l_words)# 搜索的症状对应的idf
                m_p = m_data * v_idf  #  各疾病加权后的p
                if len(l_words) == 1:
                    v_idf = np.hstack((v_idf, 1))
                    m_p = np.hstack((m_p, np.ones((len(l_ill), 1))))
                final_p = v_idf.dot(m_p.T) / ( np.sqrt(np.sum(v_idf ** 2)) * np.sqrt(np.sum(m_p ** 2, axis=1)))  # cos
                # print(final_p)
                # 返回结果
                idx = np.argsort(-final_p)
                for i in idx:
                    d = {}
                    d['ill'] = l_ill[i]
                    d['p'] = final_p[i]
                    icds.append(d)
                result = {"ICDS" : icds, "rulers" : symptom}
            elif method == "probability":
                l_words = []
                l_p = []
                for kw in queryWords:
                    sql_inner += "'"+ kw['keyword'].encode('utf-8') +"',"
                    l_words.append(kw['keyword'])  # 症状
                    l_p.append(kw['p'])  #  p 值
                sql = "select * from DOC_Rating_P where keyword in("+ sql_inner.rstrip(',') +") "
                r = db.select(sql)
                l_ill = list(set([x[1] for x in r]))  # 疾病
                m_data = np.zeros((len(l_ill), len(l_words)))
                for row in r:
                    m_data[l_ill.index(row[1])][l_words.index(row[0])] = row[2]
                l_result = []
                for row in m_data:
                    x = 1
                    for c in row:
                        x = x * (1 - c)
                    row_r = 1 - x
                    print(row_r)
                    l_result.append(row_r)
                idx = np.argsort(-np.array(l_result))
                for i in idx:
                    d = {}
                    d['ill'] = l_ill[i]
                    d['p'] = l_result[i]
                    d['ICD'] = self.ill_icd[l_ill[i]]
                    icds.append(d)
                result = {"ICDS" : icds, "rulers" : symptom}
            elif method == "original":
                response = []
                icd_list = []
                result = self.search(condition, subject, is_penalize=is_penalize)
                for row in result['data']:
                    if row['ICD'] not in icd_list:
                        icd_list.append(row['ICD'])
                        icds.append({ 'ICD' : row['ICD'], 'ill':row['ill'], 'p': row['R']})
                result = {"ICDS" : icds, "rulers" : symptom}
            elif method == "distance":
                # 返回 近似病例
                if is_return_emrs:
                    emrs_cnt = defaultdict(int)
                    result = self.search(condition, subject, is_penalize=is_penalize)
                    for row in result['data']:
                        emrs_cnt[row["ICD"]] += 1
                        if emrs_cnt[row["ICD"]] == 1:
                            emrs[row["ICD"]] = [row["PATNO"]]
                        else:
                            if emrs_cnt[row["ICD"]] <= 20:
                                emrs[row["ICD"]].append(row["PATNO"])
                # 诊断
                l_words = []
                l_p = []
                for kw in queryWords:
                    sql_inner += "'"+ kw['keyword'].encode('utf-8') +"',"
                    l_words.append(kw['keyword'])  # 症状
                    l_p.append(kw['p'])  #  p 值
                sql = "select * from V_DOC_Rating_P where keyword in("+ sql_inner.rstrip(',') +") "
                r = db.select(sql)
                l_ill = list(set([x[1] for x in r]))  # 疾病
                m_data = np.zeros((len(l_ill), len(l_words)))
                for row in r:
                    m_data[l_ill.index(row[1])][l_words.index(row[0])] = row[2]

                _idf = np.array(l_p) * len(l_words)
                v_idf = _idf * 1 / _idf  # 去掉 频率的影响，并
                m_idf = m_data

                distance = np.sqrt(np.sum((m_idf - v_idf) ** 2, axis=1))
                idx = np.argsort(distance)
                for i in idx:
                    d = {}
                    d['ill'] = l_ill[i]
                    d['p'] = distance[i]
                    d['ICD'] = self.ill_icd[l_ill[i]]
                    if is_return_emrs:
                        d['EMRS'] = emrs[d['ICD']]
                    icds.append(d)

                result = {"ICDS" : icds, "rulers" : symptom}
        return result

# _*_ coding:utf-8 _*_
import codecs
import os
import numpy as np
import CRFPP

__author__ = 'T'


# 执行批处理
def exec_cmd(action, iter=1000):
    path_base = r'E:\PythonWB\twi\crf\data'
    # learn
    crf_learn = path_base + r'\CRF++-0.58_win\crf_learn.exe -t -m {0} '.format(iter)
    crf_learn_templ =  path_base + r'\template.txt'
    crf_learn_corpora =  path_base + r'\pku_training_taged.txt'
    crf_model = path_base + r'\crf_model.txt'
    # test
    crf_test = path_base + r'\CRF++-0.58_win\crf_test.exe '
    crf_test_corpora = path_base + r'\icwb2-data\testing\pku_test.utf8'
    crf_test_corpora_res = path_base + r'\pku_test_res.utf8'

    if action == 'crf_learn':
        cmd = '{0} {1} {2} {3}'.format(crf_learn, crf_learn_templ, crf_learn_corpora, crf_model)
        os.system(cmd)
    elif action == 'crf_test':
        cmd = '{0} {1} {2} > {3}'.format(crf_test, crf_model, crf_test_corpora, crf_test_corpora_res)
        os.system(cmd)
    else:
        print 'firt paramter is illegal'


# 转化文本并添加标注
def generate_data_for_crf(file_input,file_output):
    f_in = codecs.open(file_input, 'r', 'utf-8')
    f_out = codecs.open(file_output, 'w', 'utf-8')
    for line in f_in.readlines():
        words = line.strip().split()
        for w in words:
            if len(w) == 1:
                f_out.write(w + '\tS\n')
            else:
                f_out.write(w[0] + '\tB\n')
                for i in range(1, len(w)-1):
                    f_out.write(w[i] + '\tM\n')
                f_out.write(w[-1] + '\tE\n')
        f_out.write('\n')
    f_in.close()
    f_out.close()


# crf 分词器
def crf_segmenter(input_file, output_file, tagger):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        tagger.clear()
        for word in line.strip():
            word = word.strip()
            if word:
                tagger.add((word + "\to\tB").encode('utf-8'))
        tagger.parse()
        size = tagger.size()
        xsize = tagger.xsize()
        for i in range(0, size):
            for j in range(0, xsize):
                char = tagger.x(i, j).decode('utf-8')
                tag = tagger.y2(i)
                if tag == 'B':
                    output_data.write(' ' + char)
                elif tag == 'M':
                    output_data.write(char)
                elif tag == 'E':
                    output_data.write(char + ' ')
                else: # tag == 'S'
                    output_data.write(' ' + char + ' ')
        output_data.write('\n')
    input_data.close()
    output_data.close()

# 读取一行并清空换行和空格
def read_line(f):
    line = f.readline()
    line = line.strip('\n').strip('\r').strip(' ')
    while (line.find('  ') >= 0):
        line = line.replace('  ', ' ')
    return line

# 分词评价
def test_accuracy(file_gold, file_tag):
    file_gold = codecs.open(file_gold, 'r', 'utf-8')
    file_tag= codecs.open(file_tag, 'r', 'utf-8')
    line1 = read_line(file_gold)
    N_count = 0
    e_count = 0
    c_count = 0
    e_line_count = 0
    c_line_count = 0

    while line1:
        line2 = read_line(file_tag)

        list1 = line1.split(' ')
        list2 = line2.split(' ')

        count1 = len(list1)   # 标准分词数
        N_count += count1
        if line1 == line2:
            c_line_count += 1
            c_count += count1
        else:
            e_line_count += 1
            count2 = len(list2)

            arr1 = []
            arr2 = []

            pos = 0
            for w in list1:
                arr1.append(tuple([pos, pos + len(w)]))
                pos += len(w)

            pos = 0
            for w in list2:
                arr2.append(tuple([pos, pos + len(w)]))
                pos += len(w)

            for tp in arr2:
                if tp in arr1:
                    c_count += 1
                else:
                    e_count += 1

        line1 = read_line(file_gold)

    R = c_count * 100. / N_count
    P = c_count * 100. / (c_count + e_count)
    F = 2. * P * R / (P + R)
    ER = 1. * e_count / N_count

    print '  标准词数：{} 个，正确词数：{} 个，错误词数：{} 个'.format(N_count, c_count, e_count).decode('utf8')
    print '  标准行数：{}，正确行数：{} ，错误行数：{}'.format(c_line_count+e_line_count, c_line_count, e_line_count).decode('utf8')
    print '  Recall: {}%'.format(R)
    print '  Precision: {}%'.format(P)
    print '  F MEASURE: {}%'.format(F)
    print '  ERR RATE: {}%'.format(ER)
    file_gold.close()
    file_tag.close()



if __name__ == "__main__":
    input = r'data\icwb2-data\testing\pku_test.utf8'
    output2 = r'data/pku_test.txt'
    crf_model = r'data/crf_model'
    triger = CRFPP.Tagger('-m '+ crf_model )
    crf_segmenter(input, output2, triger)

    # generate_data_for_crf(input, output)
    # translate_data_from_crf(output, translate)

    #exec_cmd('crf_learn',10)
    # exec_cmd('crf_test',10)
    """
    crf_test_corpora_gold = r'data\icwb2-data\gold\pku_test_gold.utf8'
    test_res = 'data/out_crf.txt'
    test_accuracy(crf_test_corpora_gold, test_res)
    """

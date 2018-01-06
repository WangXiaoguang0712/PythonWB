# _*_ coding:utf-8 _*_
import codecs
import os
import CRFPP

__author__ = 'T'

class CRF_Ultimate():
    def __init__(self, crf_model='data/crf_model', n_iter=1000, corpora_taged='data/corpora_taged.txt'):
        self.crf_model = crf_model
        self.n_iter = n_iter
        self.corpora_taged = corpora_taged

    # 转化文本并添加标注
    def _generate_data_for_crf(self, file_input, file_output):
        print 'begin tag for source file'
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
        print 'end tag for source file'

    # fit
    def learn(self, crf_learn, train_templ, train_corpora):
        if crf_learn == '':
            raise 'please input file full path of crf_learn.exe'
            return
        if train_templ == '':
            raise 'you should input correct feature template file'
            return
        learn_param = ' -t -m {0} -f 1 -p 2 -c 1.5 -a CRF-L2 '.format(self.n_iter)
        # -a CRF-L2 or CRF-L1  :规范化算法选择 一般来说L2算法效果要比L1算法稍微好一点，虽然L1算法中非零特征的数值要比L2中大幅度的小。
        # -c float:拟合度 c的数值越大，CRF拟合训练数据的程度越高
        # -m:循环次数
        # -f NUM: 这个参数设置特征的cut-off threshold,至少NUM次出现的特征。默认值为1
        # -p 线程数量，适用多CPU
        self._generate_data_for_crf(train_corpora,self.corpora_taged)
        cmd = '{0} {1} {2} {3} {4}'.format(crf_learn, learn_param, train_templ, self.corpora_taged, self.crf_model)
        os.system(cmd)

    # predict
    def segment(self, f_input, f_output):
        tagger = CRFPP.Tagger('-m ' + self.crf_model)
        input_data = codecs.open(f_input, 'r', 'utf-8')
        output_data = codecs.open(f_output, 'w', 'utf-8')
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
    def _read_line(self,f):
        line = f.readline()
        line = line.strip('\n').strip('\r').strip(' ')
        while (line.find('  ') >= 0):
            line = line.replace('  ', ' ')
        return line

    # test accuracy
    def test_accuracy(self, f_output, f_output_std):
        fs_output_std = codecs.open(f_output_std, 'r', 'utf-8')
        fs_output= codecs.open(f_output, 'r', 'utf-8')
        line1 = self._read_line(fs_output_std)
        N_count = 0
        e_count = 0
        c_count = 0
        e_line_count = 0
        c_line_count = 0
        while line1:
            line2 = self._read_line(fs_output)
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
            line1 = self._read_line(fs_output_std)
        R = c_count * 100. / N_count
        P = c_count * 100. / (c_count + e_count)
        F = 2. * P * R / (P + R)
        ER = 100. * e_count / N_count
        print '  标准词数：{} 个，正确词数：{} 个，错误词数：{} 个'.format(N_count, c_count, e_count).decode('utf8')
        print '  标准行数：{}，正确行数：{} ，错误行数：{}'.format(c_line_count+e_line_count, c_line_count, e_line_count).decode('utf8')
        print '  Recall: {}%'.format(R)
        print '  Precision: {}%'.format(P)
        print '  F MEASURE: {}%'.format(F)
        print '  ERR RATE: {}%'.format(ER)
        fs_output_std.close()
        fs_output.close()


if __name__ == "__main__":
    path_base = r'E:\PythonWB\twi\crf\data'
    crf_learn = path_base + r'\CRF++-0.58_win\crf_learn.exe'
    crf_learn_corpora =  path_base + r'\icwb2-data\training\pku_training.utf8'
    crf_input = path_base + r'\icwb2-data\testing\pku_test.utf8'
    crf_output_std = path_base + r'\icwb2-data\gold\pku_test_gold.utf8'

    crf_learn_corpora_taged =  path_base + r'\pku_training_taged.txt'
    crf_learn_templ =  path_base + r'\template.txt'
    crf_model = path_base + r'\crf_model'
    crf_output = path_base + r'\pku_test.txt'

    model = CRF_Ultimate(n_iter=10, crf_model=crf_model, corpora_taged=crf_learn_corpora_taged)
    # model.learn(crf_learn, crf_learn_templ, crf_learn_corpora)
    # model.segment(crf_input, crf_output)
    model.test_accuracy(crf_output, crf_output_std)

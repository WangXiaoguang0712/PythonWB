#coding:utf-8
__author__ = 'T'

import re
import numpy as np
import pandas as pd
from matplotlib import  pyplot as plt
# from statsmodels.tsa import stattools as stt

file_path = '../data/wk_ill_desire.xlsx'
df_data = pd.read_excel(file_path,sheet_name='Sheet1',header=0)

l = {}
for ill in df_data.ix[:,1].drop_duplicates():
    ill_data = df_data[df_data.ix[:,1] == ill]
    ill_std = np.std(ill_data['fee_exa']/ill_data['fee_all'])
    l[ill] = ill_std
print(l)
#s = sorted(l.iteritems(), key=lambda x:x[1], reverse=True)
#print s

for k,v in l.items():
    print('%s => %s' %(k,v))




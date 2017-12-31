# -*- coding:utf-8 -*-
__author__ = 'T'
import numpy as np
import pandas as pd
import matplotlib as mpl
#import matplotlib.pyplot as plts
#arry
arr = [1,3,4,5]
#print arr

#Series
s = pd.Series([1,3,5,np.nan,6,8])

#dateframe
dates = pd.date_range('20130101',periods=6)
#print dates
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
#print np.random.randn(6,4)
df2 = pd.DataFrame({'A':1,
                    'B':pd.Timestamp('20130103'),
                    'C':pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D':np.array([3]*4,dtype='int32'),
                    'E':pd.Categorical(["test","train","test","train"]),
                    'F':'f00'})
print df2
#print df2
#print pd.Series(1,index=list(range(4)),dtype='float32')
#print np.array([3]*4,dtype='int32'),
#print df2.dtypes

#print df.head()
#print df.tail(3)
#print df.index
#print df.values
#print df.describe()
#print df.T
#print df.sort_index(axis=0,ascending=False)
#print df.sort(columns='B')

#print df['A']+df.A
#print df.loc[dates[0]]
#print df.loc[:,['A','B']]
#print df.loc['20130101',['A','B']]
#print df.iloc[3]
#print df[df.A>0]
#print df[df>0]

#df.at[dates[0],'A']=0
#df.iat[2,2]=0
#print df

#df2 = df.copy()
#df2['E']=['one','two','three','four','one','three']
#print df2
#print df2[df2['E'].isin(['one','three'])]

"""
s1 = pd.Series([1,2,3,4,5,6],index=pd.date_range('20130102',periods=6))
df['F']=s1

df.at[dates[0],'A']=0
df.iat[0,1]=0
df.loc[:,'D']=np.array([5]*len(df))
print df


df1 = df.reindex(index=dates[0:4],columns=list(df.columns)+['E'])
#df1.loc[dates[0]:dates[1],'E']=1
df1.loc[0:2,'E']=1
#df1.dropna(how='any')
df1.fillna(value=5)
print pd.isnull(df1)

#统计
#print df.mean(1)

s = pd.Series([1,3,5,np.nan,6,8],index=dates).shift(2)
print s
print df.sub(s,axis='index')


a=np.arange(1,10,1)
print df.apply(np.cumsum)

#print
df = pd.DataFrame(np.random.randn(10,4))
se =[df[:3],df[3:7],df[7:]]
print pd.concat(se)
"""
"""
#合并
left = pd.DataFrame({'key':['foo','foo'],'lval':[1,2]})
right = pd.DataFrame({'key':['foo','foo'],'lval':[4,5]})
center = pd.merge(left,right,on='key')

print center.append(left)
"""
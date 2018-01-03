#coding:utf-8
__author__ = 'T'

import numpy as np
import pandas as pd

class Outlier_dectector(object):
    def __init__(self,is_week = False):
        self.is_week = is_week
        self.n_sima = 1.96
        self.n_quantile = 1.5

    def loaddata(self):
        file_path = 'data/hspdata.xlsx'
        df_data = pd.read_excel(file_path,sheet_name='Sheet2',header=0)
        df_data['B'] = df_data['B'].astype(str)
        df_data_pivot = df_data.pivot(index='B',columns='A',values='C')
        df_data_pivot.index = pd.DatetimeIndex(df_data_pivot.index)
        #print df_data_pivot.info()
        return df_data_pivot

    def generate_estimatior(self,p_type,data):
        if p_type == 1:
            fm,fs = np.mean(data),np.std(data)
            estimator_min, estimator_max = (fm - self.n_sima * fs,fm + self.n_sima * fs)
        elif p_type == 2:
            fq1,fq3 = data.quantile(0.25),data.quantile(0.75)
            estimator_min, estimator_max = (fq1 - self.n_quantile * (fq3 - fq1),fq3 + self.n_quantile * (fq3 - fq1))
        else:
            raise(ValueError,'not support p_type!')
        return estimator_min, estimator_max

    def analyze_data(self,is_week,p_type,data_hsp):
        data_hsp_outlier = pd.Series()
        if is_week:
            for j in range(7):
                data_hsp_week = data_hsp[map(lambda x:x.weekday() == j,pd.to_datetime(data_hsp.index))]
                fm,fs = np.mean(data_hsp_week),np.std(data_hsp_week)
                estimator_min,estimator_max = self.generate_estimatior(p_type,data_hsp_week)
                data_hsp_outlier = data_hsp_outlier.append(data_hsp_week[data_hsp_week > estimator_max].map(lambda x:(x,'+')))
                data_hsp_outlier = data_hsp_outlier.append(data_hsp_week[data_hsp_week < estimator_min].map(lambda x:(x,'-')))
                data_hsp_outlier = data_hsp_outlier.rename(data_hsp.name)
        else:
            estimator_min,estimator_max = self.generate_estimatior(p_type,data_hsp)
            data_hsp_outlier = data_hsp_outlier.append(data_hsp[data_hsp > estimator_max].map(lambda x:(x,'+')))
            data_hsp_outlier = data_hsp_outlier.append(data_hsp[data_hsp < estimator_min].map(lambda x:(x,'-')))
            data_hsp_outlier = data_hsp_outlier.rename(data_hsp.name)
        return data_hsp_outlier

    def detect_by_normaldistribution(self):
        data = self.loaddata()
        df = pd.DataFrame()
        for i in range(len(data.columns)):
            data_hsp = data.ix[:,i]
            data_hsp_outlier = self.analyze_data(self.is_week,1,data_hsp)

            if len(data_hsp_outlier) > 0:
                if len(df) == 0:
                    df = pd.DataFrame(data_hsp_outlier)
                else:
                    df = pd.merge(df,pd.DataFrame(data_hsp_outlier),how='outer',left_index=True,right_index=True)
        return df


    def detect_by_quantile(self):
        data = self.loaddata()
        df = pd.DataFrame()
        for i in range(len(data.columns)):
            data_hsp = data.ix[:,i]
            data_hsp_outlier = self.analyze_data(self.is_week,2,data_hsp)

            if len(data_hsp_outlier) > 0:
                if len(df) == 0:
                    df = pd.DataFrame(data_hsp_outlier)
                else:
                    df = pd.merge(df,pd.DataFrame(data_hsp_outlier),how='outer',left_index=True,right_index=True)
        return df

if __name__ == "__main__":
    detector = Outlier_dectector()

    wr = pd.ExcelWriter('hsp_data_err.xlsx')
    df1 = detector.detect_by_normaldistribution()
    df1.to_excel(wr,sheet_name='by_normal')
    df = detector.detect_by_quantile()
    df.to_excel(wr,sheet_name='by_quantile')
    wr.save()
    wr.close()

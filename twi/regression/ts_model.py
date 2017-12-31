# _*_coding:utf-8 _*_

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa import stattools as stt
from statsmodels.graphics import tsaplots
from statsmodels.tsa import seasonal as ssl
from statsmodels.tsa import arima_model as am
import time
__author__ = 'T'


def to_date(dates):
    return pd.datetime.strptime(dates, '%Y-%m')


def load_data():
    ts_data = pd.read_csv('data/AirPassengers.csv', parse_dates=['Month'], index_col='Month', date_parser=to_date)
    # data = pd.read_csv('data/AirPassengers.csv',index_col='Month')
    return ts_data['#Passengers']

def load_data_hsp():
    ts_data = pd.read_excel('data/hspdata.xlsx', sheet_name='Sheet3',index_col='A', header= 0,parse_dates=['A'],
                            date_parser=lambda x:pd.datetime.strptime(x,'%Y%m%d'))
    return ts_data['B']

class TsModel(object):
    def __init__(self, ts, n_test = 7, window = 12,span='day'):
        self.window = window
        self.span = span
        # 分训练集和测试集
        self.ts_train = ts[:- n_test]
        self.ts_test = ts[- n_test:]
        ts_log_diff1 = None
        ts_log_diff2 = None
        ts_log_ma_sub = None
        self.ts_stable = None
        self.ts_dict = {'ts_log_diff1':ts_log_diff1, 'ts_log_diff2':ts_log_diff2, 'ts_log_ma_sub':ts_log_ma_sub,
                         'ts':ts}

    def test_stationarity(self, time_series, isplot=False):
        # Determing rolling statistics
        rol_mean = time_series.rolling(window=self.window).mean()
        rol_std = time_series.rolling(window=self.window).std()
        if isplot:
            # plot
            plt.figure()
            plt.plot(time_series, color='b', label='original')
            plt.plot(rol_mean, color='r', label='rolling mean')
            plt.plot(rol_std, color='black', label='rolling std')
            plt.legend(loc='best')
            plt.title("Rolling Mean & Standard Deviation")
            # plt.show(block=False)
            plt.show()
        # df 测试
        print 'Result of Dicky-Fuller test'
        dftest = stt.adfuller(time_series, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical value(%s)' % key] = value
        print dfoutput

    def test_whitenoise(self, ts, nlag):
        from statsmodels.stats.diagnostic import acorr_ljungbox
        print('whitenoisy:%s' % acorr_ljungbox(ts, nlag))

    def check_data(self,temp_ts):
        print('>>>>>>>>>> ts')
        self.test_stationarity(temp_ts)
        print('>>>>>>>>>> ts_log_diff1')
        ts_log = np.log(temp_ts)
        self.ts_dict['ts_log_diff1'] = ts_log.diff().dropna()
        self.test_stationarity(self.ts_dict['ts_log_diff1'])
        print('>>>>>>>>>> ts_log_diff2')
        self.ts_dict['ts_log_diff2'] = self.ts_dict['ts_log_diff1'].diff().dropna()
        self.test_stationarity(self.ts_dict['ts_log_diff2'])
        print('>>>>>>>>>> ts_log_ma_sub')
        ts_log_ma = ts_log.rolling(window=self.window).mean()
        self.ts_dict['ts_log_ma_sub'] = (ts_log - ts_log_ma.shift()).dropna()
        self.test_stationarity(self.ts_dict['ts_log_ma_sub'])


    # 显示 自相关系数与骗自相关系数
    def show_acf_pacf(self,tag):
        try:
            self.ts_stable = self.ts_dict[tag]
        except:
            raise ValueError('not support tag value')
        # acf & pacf
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        tsaplots.plot_acf(self.ts_stable, lags=40, ax=ax1)
        tsaplots.plot_pacf(self.ts_stable, lags=40, ax=ax2)
        plt.show()

    def desire_pdq(self, tag):
        self.ts_stable = self.ts_dict[tag]
        bic_matrix = []
        for p in range(7 + 1):
            tmp = []
            for q in range(4 + 1):
                try:
                    model_res = am.ARIMA(self.ts_stable, order=(p,0,q)).fit(disp=-1)
                    # tmp.append(sum((model_res.fittedvalues - self.ts_stable) ** 2))
                    tmp.append(model_res.bic)
                except:
                    tmp.append(None)
            bic_matrix.append(tmp)
        df_bic = pd.DataFrame(bic_matrix)
        print df_bic.stack()

    def test_arima_model(self,tag):
        try:
            self.ts_stable = self.ts_dict[tag]
        except:
            raise ValueError('not support tag value')
        print('i am in processing')
        print self.ts_stable
        model = am.ARIMA(self.ts_stable, order=(6, 0, 1))
        result_ARIMA = model.fit(disp=- 1)
        plt.plot(self.ts_stable, label=tag)
        plt.plot(result_ARIMA.fittedvalues, color='red', label='result_AR.fittedvalues')
        plt.legend(loc='best')
        plt.title('MA model SSE:%.4f' % (sum((result_ARIMA.fittedvalues - self.ts_stable) ** 2)))
        plt.show()

    def test_predict(self, tag, n=5):
        self.check_data(self.ts_train)
        try:
            self.ts_stable = self.ts_dict[tag]
        except:
            raise ValueError('not support tag value')
            return 0
        print('i am in processing {0}'.format(tag))

        ts_log = np.log(self.ts_train)

        if tag == 'ts_log_ma_sub':
            ts_log_ma = ts_log.rolling(window=self.window).mean()
            model = am.ARIMA(self.ts_stable, order=(12, 0, 0))
            rs_of_arima = model.fit(disp=- 1)

            predict_val = rs_of_arima.forecast(n)[0]
            fit_ts_log_ma_sub = rs_of_arima.fittedvalues  # 拟合值
            fit_ts_log = fit_ts_log_ma_sub.add(ts_log_ma.shift(), fill_value=0)  # 拟合值 减均值还原
            for i in range(1,n + 1):
                if self.span == 'day':
                    newindex = (ts_log_ma.index[-1] + pd.tseries.offsets.DateOffset(months=1, days=0))  # 新索引
                elif self.span == 'month':
                    newindex = (ts_log_ma.index[-1] + pd.tseries.offsets.DateOffset(months=1, days=0))  # 新索引

                fit_ts_log[newindex] = predict_val[i - 1] + ts_log_ma[-1]  # 预测值 +  原移动均值最后一位 = 减均值还原
                ts_log_ma[newindex] = np.mean(fit_ts_log[-self.window:-1])  # 用新预测序列计算移动均值，存入原移动均值序列，备下次使用

            fit_ts = np.exp(fit_ts_log)  # log 还原
            all_ts = self.ts_train.append(self.ts_test)  # 合并训练集与测试集

            plt.plot(all_ts, color='grey', linestyle='-', label='ts')
            plt.plot(fit_ts[:- n], color='b', linestyle='-', label='predict_ts')
            plt.plot(fit_ts[- n - 1:], color='r', linestyle='-', label='predict_ts')
            plt.title('RMSE of arima:{0}'.format(np.sqrt(np.mean((fit_ts - all_ts) ** 2))))
            plt.legend(loc=('best'))
            plt.show()
        elif tag == 'ts_log_diff1':
            model = am.ARIMA(self.ts_stable, order=(6, 0, 1))
            rs_of_arima = model.fit(disp=- 1)
            # 预测值 n 期
            predict_val = rs_of_arima.forecast(n)[0]
            fit_ts_log_diff1 = rs_of_arima.fittedvalues  # 拟合值
            fit_ts_log = rs_of_arima.fittedvalues + ts_log.shift()  # 拟合值还原
            for i in range(1,n + 1):
                if self.span == 'day':
                    newindex = (ts_log.index[-1] + pd.tseries.offsets.DateOffset(months=0, days=1))
                elif self.span == 'month':
                    newindex = (ts_log.index[-1] + pd.tseries.offsets.DateOffset(months=1, days=0))
                fit_ts_log[newindex] = predict_val[i - 1] + ts_log[-1]  # 预测值 +  原序列最后一位 = 预测值还原
                ts_log[newindex] = fit_ts_log[newindex]  # 在原序列后用预测序列填充

            fit_ts = np.exp(fit_ts_log)  # log 还原
            all_ts = self.ts_train.append(self.ts_test)  # 合并训练集与测试集
            # 累积误差
            # predict_ts_log2 = pd.Series(ts_log.ix[0], index=ts_log.index).add(fit_ts_log_diff1.cumsum(), fill_value=0)
            # predict_ts2 = np.exp(predict_ts_log2)
            # x_tick = map(lambda x:time.strftime('%Y-%m-%d',time.strptime(str(x),'%Y-%m-%d %H:%M:%S')),all_ts.index)
            plt.plot(all_ts,color='grey', linestyle='-', marker='o', label='all_ts')
            plt.plot(fit_ts[:-n],color='b', linestyle='-', marker='o', label='fit_ts')
            plt.plot(fit_ts[- n - 1:], color='r', linestyle='-', marker='o', label='fit_ts_predict')
            # plt.plot(predict_ts2,color='g',linestyle='-',label='predict_ts2')
            plt.title('RMSE of arima:{0}'.format(round(np.sqrt(np.mean((fit_ts - all_ts) ** 2)), 4)))
            # plt.xticks(x_tick,rotation=45)
            plt.show()

        print fit_ts.tail()
    #
    def handy(self, tag, ischeckdata, iscf, isqdp, act):
        if ischeckdata == True:
            self.check_data(self.ts_train.append(self.ts_test))
        if iscf == True:
            self.check_data(self.ts_train)
            self.show_acf_pacf(tag)
        if isqdp == True:
            self.check_data(self.ts_train)
            self.desire_pdq(tag)
        if act == 1:
            self.check_data(self.ts_train)
            self.test_arima_model(tag)
        elif act == 2:
            self.test_predict(tag, n=5)
        elif act == 0:
            pass
        else:
            raise ValueError('not support args')

if __name__ == "__main__":
    # data = load_data_hsp()
    data = load_data()
    tsm = TsModel(data, n_test=3, window=9, span='day')
    tsm.handy('ts_log_diff1',False,False,False,2)
    #tsm.arima_predict()
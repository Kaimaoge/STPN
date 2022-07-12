import numpy as np
import copy
from statsmodels.tsa.vector_ar.var_model import VAR

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def test_error(y_predict, y_test):
    """
    Calculates MAE, RMSE, R2.
    :param y_test:
    :param y_predict.
    :return:
    """
    err = y_predict - y_test
    MAE = np.mean(np.abs(err[~np.isnan(err)]))
    
    s_err = err**2
    RMSE = np.sqrt(np.mean((s_err[~np.isnan(s_err)])))
    
    test_mean = np.mean((y_test[~np.isnan(y_test)]))
    m_err = (y_test - test_mean)**2
    R2 = 1 - np.sum(s_err[~np.isnan(s_err)])/np.sum(m_err[~np.isnan(m_err)])
    
    return MAE, RMSE, R2
    

def historical_average_predict(np_, period= 2*18 , test_ratio=0.2):
    """
    Calculates the historical average of route delay.
    :param np:
    :param period: default 1 day.
    :param test_ratio:
    :return:
    """
    n_route, n_sample = np_.shape 
    n_test = int(round(n_sample * test_ratio)) 
    n_train = n_sample - n_test  
    y_test = np_[:,-n_test:]
    y_predict = copy.deepcopy(y_test)
    
    for i in range(n_train, min(n_sample, n_train + period)):
        inds = [j for j in range(i % period, n_train, period)]
        historical = np_[:, inds]
        for k in range(n_route):
            y_predict[k, i - n_train] = historical[k, :][~np.isnan(historical[k, :])].mean()   
    for i in range(n_train + period, n_sample, period):
        size = min(period, n_sample - i)
        start = i - n_train
        y_predict[:, start:start + size] = y_predict[:, start - period: start + size - period]
    
    return y_predict, y_test

def var_predict(np_, n_forwards=(1,2,3,4,5,6,7,8,9,10,11,12), n_lags=36, test_ratio=0.2):
    """
    Multivariate time series forecasting using Vector Auto-Regressive Model.
    :param df: numpy, route x time.
    :param n_forwards: a tuple of horizons.
    :param n_lags: the order of the VAR model.
    :param test_ratio:
    :return: [list of prediction in different horizon], dt_test
    """
    n_route, n_sample = np_.shape 
    n_test = int(round(n_sample * test_ratio)) 
    n_train = n_sample - n_test
    df_train, df_test = np_[:, :n_train], np_[:, n_train:]
    mean_train = np.mean(df_train[~np.isnan(df_train)])
    std_train = np.std(df_train[~np.isnan(df_train)])
    scaler = StandardScaler(mean=mean_train, std=std_train)
    data = scaler.transform(df_train)
    data[np.isnan(data)] = 0
    data = data.T
    var_model = VAR(data)
    var_result = var_model.fit(n_lags)
    max_n_forwards = np.max(n_forwards)
    result = np.zeros(shape=(len(n_forwards), n_test, n_route))
    start = n_train - n_lags - max_n_forwards + 1
    for input_ind in range(start, n_sample - n_lags):
        inputs = scaler.transform(np_[:,input_ind: input_ind + n_lags].T)
        inputs[np.isnan(inputs)] = 0
        prediction = var_result.forecast(inputs, max_n_forwards)
        for i, n_forward in enumerate(n_forwards):
            result_ind = input_ind - n_train + n_lags + n_forward - 1
            if 0 <= result_ind < n_test:
                result[i, result_ind, :] = prediction[n_forward - 1, :]
        
    return scaler.inverse_transform(result), df_test

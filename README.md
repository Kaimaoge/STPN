# Spatiotemporal Propagation Learning for Network-Wide Flight Delay Prediction

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)
[![GitHub stars](https://img.shields.io/github/stars/Kaimaoge/STPN.svg?logo=github&label=Stars&logoColor=white)](https://github.com/Kaimaoge/STPN)

This is the code corresponding to the experiments conducted for our paper ["Spatiotemporal Propagation Learning for Network-Wide Flight Delay Prediction"](https://arxiv.org/abs/2207.06959)

About this Project
--------------
We develop a space-time separable multi-graph convolutional network for learning airport network delay propagation patterns. The model is applied to multi-step ahead arrival and departure delay prediction. The aim of this project relates to a fundamental question in spatiotemporal modeling -- **how to accurately and efficiently model spatiotemporal dependencies**. 

Dataset
--------------
The data used in this paper can be downloaded from [Baidu drive](https://pan.baidu.com/s/13siqq4ffpxhvootkJKvgbw)(code: vz59). 

Tasks and Challenges
--------------

<p align="center">
<img align="middle" src="https://github.com/Kaimaoge/STPN/blob/main/image/delay_propagation.png" width="600" />
</p>

This example shows that the complexity of spatiotemporal dependencies within airport network delay lies in the following four aspects: 1) Exogenous Factors (powerful snow in this figure), 2) Multi-relational Spatial Dependencies, 3) Coupled Spatiotemporal Effects, 4) Departure-Arrival Delay Relationship.


### Model implementation

<p align="center">
<img align="middle" src="https://github.com/Kaimaoge/STPN/blob/main/image/framework_all.png" width="700" />
</p>

The proposed model leverages the spatiotemporal propagation patterns learned from historical departure/arrival delays and external factors (weather conditions) to forecast the long-term future departure/arrival delays. The propagation patterns are learned by space-time separable multi-graph convolution, which considers the joint space-time interaction between airports. The given figure illustrates the basic building block of STPN. The effects of geographic proximity, weather conditions, and traffic volume on delay propagation are incorporated into STPN.

Example commands for running our model
--------------

To use the code, please first obtain the datasets from [Baidu drive](https://pan.baidu.com/s/13siqq4ffpxhvootkJKvgbw)(code: vz59). Then place China and US datasets in cdata and udata respectively. The original datasets are collected from [U.S. Bureau of Transportation Statistics](https://www.transtats.bts.gov) and [Xiecheng](https://pan.baidu.com/s/1dEPyMGh#list/path=\%2F).

We can train a U.S airport delay forecastor as following:

```bash
python training_u.py \
			--train_val_ratio 0.6 0.3 --use_se False \
			--heads 2
```
Then, we use 60% data as training data, 30% data as validation data, and 10% as test data. For the model configuration, we do not use SE-block for feature importance recalibration, and we only use 2 heads self-attention for temporal attention learning.

You can directly run 
```bash
python training_u.py
```
This is the default training methods we used in our paper.

For testing
```bash
python test_u.py --in_len 6
```
You can change the length of input sequence, since we use a self-attention model to learn temporal dependencies. The length of the inputs does not significantly influence our model.

Additional results on forecasting traffic flow
--------------

STPN also is a competitive forecastor for general spatiotemporal forecasting task.

| Methods | SVR | LSTM | STGCN | Gwave | STFGNN | AGCRN | STPN |
|-----------------------------|----------------------|-----------------------|------------------------|------------------------|-------------------------|------------------------|-----------------------|
| Datasets                    | Metrics              |                       |                        |                        |                         |                        |                       |                  |
|     | MAE                  | 21.07                 | 21.33                  | 17.49                  | 19.85                   | 16.77                  | 15.98                 | 15.74} |
|     PeMS03                         | MAPE                 | 21.07                 | 23.33                  | 17.15                  | 19.31                   | 16.30                  | 15.23      | 16.40            |
|                             | RMSE                 | 35.29                 | 35.11                  | 30.12                  | 19.31                   | 28.34                  | 28.25                 | 24.63 |
|       | MAE                  | 28.70                 | 27.14                  | 22.70                  | 25.45                   | 19.83       | 19.83      | 20.38            |
|     PeMS04                        | MAPE                 | 19.20                 | 18.20                  | 14.59                  | 17.29                   | 13.02                  | 12.97      | 13.28            |
|                             | RMSE                 | 44.56                 | 41.59                  | 35.55                  | 39.70                   | 31.88       | 32.30                 | 32.52            |
|     | MAE                  | 32.49                 | 29.98                  | 25.38                  | 26.85                   | 22.07                  | 22.37                 | 22.29 |
|  PeMS07                            | MAPE                 | 14.26                 | 13.20                  | 11.08                  | 12.12                   | 9.21                   | 9.21                  | 9.61             |
|                             | RMSE                 | 50.22                 | 45.84                  | 38.78                  | 42.78                   | 35.80                  | 36.55                 | 34.97 |
|       | MAE                  | 23.25                 | 22.20                  | 18.02                  | 19.13                   | 16.64                  | 15.95      | 16.50            |
|   PeMS08                          | MAPE                 | 14.64                 | 14.20                  | 11.40                  | 12.68                   | 10.60                  | 10.09     | 10.54            |
|                             | RMSE                 | 36.16                 | 34.06                  | 27.83                  | 31.05                   | 26.22                  | 25.22     | 25.90            |



# STPN_ariport_delay

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)
[![GitHub stars](https://img.shields.io/github/stars/Kaimaoge/STPN.svg?logo=github&label=Stars&logoColor=white)](https://github.com/Kaimaoge/STPN)

About this Project
--------------
We develop a space-time separable multi-graph convolutional networks for learning airport network delay propagation patterns. The model is applied to multi-step ahead arrival and departure delay prediction. The aim of this project relates to a fundamental question in spatiotemporal modeling -- **how to accurate and efficient model spatiotemporal dependencies**.

Dataset
--------------
The data used in this paper can be downloaded from [Baidu drive](https://pan.baidu.com/s/13siqq4ffpxhvootkJKvgbw)(code: vz59). 

Tasks and Challenges
--------------

<p align="center">
<img align="middle" src="https://github.com/Kaimaoge/STPN/blob/master/image/delay.png" width="600" />
</p>

This example shows that the complexity of spatiotemporal dependencies within airport network delay lies in the following four aspects: 1) Exogenous Factors (powerful snow in this figure), 2) Multi-relational Spatial Dependencies, 3) Coupled Spatiotemporal Effects, 4) Departure-Arrival Delay Relationship.


### Model implementation

<p align="center">
<img align="middle" src="https://github.com/Kaimaoge/STPN/blob/master/image/framework.png" width="700" />
</p>

The proposed model leverages the spatiotemporal propagation patterns learned from historical departure/arrival delays and external factors (weather conditions) to forecast the long-term future departure/arrival delays. The propagation patterns are learned by space-time separable multi-graph convolution, which considers the joint space-time interaction between airports. The given figure illustrates the basic building block of STPN. The effects of geographic proximity, weather conditions, and traffic volume on delay propagation are incorporated into STPN.

Example commands for certification
--------------

To use the code, please first obtain the datasets from [Baidu drive](https://pan.baidu.com/s/13siqq4ffpxhvootkJKvgbw)(code: vz59). Then place China and US datasets in cdata and udata respectively. The original datasets are collected from [U.S. Bureau of Transportation Statistics](https://www.transtats.bts.gov/DL SelectFields.asp?gnoyr VQ=) and [Xiecheng](https://pan.baidu.com/s/1dEPyMGh#list/path=\%2F).

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

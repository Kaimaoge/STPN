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
<img align="middle" src="https://github.com/Kaimaoge/STPN/blob/master/image/delay.pdf" width="600" />
</p>

This example shows that the complexity of spatiotemporal dependencies within airport network delay lies in the following four aspects: 1) Exogenous Factors (powerful snow in this figure), 2) Multi-relational Spatial Dependencies, 3) Coupled Spatiotemporal Effects, 4) Departure-Arrival Delay Relationship.


### Model implementation

<p align="center">
<img align="middle" src="https://github.com/Kaimaoge/STPN/blob/master/image/framework.pdf" width="700" />
</p>

The proposed model leverages the spatiotemporal propagation patterns learned from historical departure/arrival delays and external factors (weather conditions) to forecast the long-term future departure/arrival delays. The propagation patterns are learned by space-time separable multi-graph convolution, which considers the joint space-time interaction between airports. The given figure illustrates the basic building block of STPN. The effects of geographic proximity, weather conditions, and traffic volume on delay propagation are incorporated into STPN.

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
<img align="middle" src="https://github.com/Kaimaoge/STPN/blob/master/images/delay.pdf" width="800" />
</p>

This example shows that the complexity of spatiotemporal dependencies within airport network delay lies in the following four aspects: 1) Exogenous Factors (powerful snow in this figure), 2) Multi-relational Spatial Dependencies, 3) Coupled Spatiotemporal Effects, 4) Departure-Arrival Delay Relationship.

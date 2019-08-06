## Constant Velocity Pedestrian Motion Prediction

This repository contains an implementation of the Constant Velocity Model from paper:

[The Simpler the Better: Constant Velocity for Pedestrian Motion Prediction](https://arxiv.org/abs/1903.07933)<br>
Christoph Schöller, Vincent Aravantinos, Florian Lay, Alois Knoll<br>
arXiv, 2019

In particular, it allows to reproduce the results for **OUR** and **OUR-S** from Table 1. The dataset in this repository is the same as the one provided [here](https://github.com/agrimgupta92/sgan), but converted to json format.

<br/>

Prediction examples of **OUR** (left) and **OUR-S** (right):

<img src="images/pred_our.png" width="40%" height="40%" align="left">
<img src="images/pred_our-s.png" width="40%" height="40%" align="center">

**NOTE**:  
We fixed minor discrepancies in the data loading, this leads to changes in the final results. We will update the manuscript with this changes. The new errors for the CVM are (ADE/FDE):  
OUR: 0.39/0.83  
OUR-S: 0.28/0.56


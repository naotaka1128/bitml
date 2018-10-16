# Bitml - bitcoin price estimator
## About This Repository
This is just a prototype to estimate the price movement of bitcoin using machine learning.

Hosting this code in Github is also evidence that we could not make money using this code.

When using this code, please take responsibility at your own risk.


## How to predict price movement
It predicts whether upside and downside price movements will occur within a certain period of time rather than movement of the price itself.

In other words, we solve the price movements of bitcoin using machine learning as a classification problem, not as a regression problem


Specifically, the 1% price falls within 1 hour (flag = 1), 1% price goes up (flag = 2), and others (flag = 0).

You can select LightGBM or CatBoost as the algorithm to use.

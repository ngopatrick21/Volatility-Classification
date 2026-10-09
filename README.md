Market Volatility Risk Classification

I built this project to automatically flag and classify market risk regimes for the S&P 500 (SPY), helping identify high-volatility shifts before they hit. 

What it does:
You feed in historical price data, and the model processes technical indicators to classify 3-day-forward market volatility into low, medium, or high-risk tiers.

How it works:
* Pulls historical pricing and volume data directly using `yfinance`.
* Engineers 5 core features, including rolling volatility, normalized volume trends, and moving-average slope indicators.
* Trains and benchmarks both a Logistic Regression baseline and an XGBoost classifier.
* Evaluates performance using precision, recall, and confusion matrix analysis to handle risk classification trade-offs.

Results:
* Test Accuracy: ~77% with the tuned XGBoost model.
* Key Takeaway: Tree-based models like XGBoost captured non-linear market velocity and volume shifts significantly better than linear baselines, giving a cleaner separation of risk regimes.

How to run it:
```bash
pip install pandas numpy scikit-learn xgboost yfinance
# Open and run the notebook:
Volatility_Classification.ipynb

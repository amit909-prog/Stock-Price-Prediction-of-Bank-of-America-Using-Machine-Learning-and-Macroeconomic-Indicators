# Stock-Price-Prediction-of-Bank-of-America-Using-Machine-Learning-and-Macroeconomic-Indicators


This project focuses on predicting the **next-day closing price of Bank of America (BAC)** using Machine Learning techniques. We used historical stock prices of major banks and macroeconomic indicators as input features. The model with the best performance was identified after evaluation using multiple regression techniques.

---

## 🧭 Table of Contents

1. [💡 What is the Goal of This Project?](#-what-is-the-goal-of-this-project)
2. [🤖 How is Machine Learning Used in Stock Prediction?](#-how-is-machine-learning-used-in-stock-prediction)
3. [📊 Models Used in This Project](#-models-used-in-this-project)
4. [📂 What Data Was Collected and Why?](#-what-data-was-collected-and-why)
5. [🧹 How Missing Values Were Handled](#-how-missing-values-were-handled)
6. [🛠️ Feature Engineering Techniques](#-feature-engineering-techniques)
7. [🔀 Train-Test Split](#-train-test-split)
8. [📈 Model Evaluation and Results](#-model-evaluation-and-results)
9. [🥇 Why the Best Model Performed Best](#-why-the-best-model-performed-best)


## 💡 What is the Goal of This Project?

💡 What is the Goal of This Project?
The goal of this project is to predict the next day's closing price of Bank of America (BAC) stock using machine learning techniques. Instead of only relying on BAC’s past prices, we build a broader view of the financial ecosystem by including:

📈 1. Peer Bank Stock Prices
Financial stocks tend to move in correlation, especially during sector-wide events.

We include lagged values (previous day's prices) from:

JPMorgan Chase (JPM)

Citigroup (C)

Morgan Stanley (MS)

Wells Fargo (WFC)

➡ These help the model understand sector-level movements that may affect BAC.



🌍 2. Macroeconomic Indicators
Macroeconomic conditions heavily influence stock prices.

We include:

^TNX (10-Year Treasury Yield) – interest rate sensitivity

^VIX (Volatility Index) – investor fear gauge

CL=F (Crude Oil Futures) – energy prices impact banks indirectly

GC=F (Gold Futures) – safe haven asset inverse relation

DX-Y.NYB (US Dollar Index) – currency strength affecting exports/imports

➡ These features help capture broader market sentiment, inflation trends, and economic risk factors.




📊 3. Technical Indicators
These help in capturing trends, momentum, and volatility in the stock price.

We compute the following for BAC stock:


✅ Moving Averages (e.g., 5-day, 10-day)

Smoothens out short-term noise to capture trend direction

Helps identify price momentum and support/resistance zones


✅ Rolling Volatility (Standard Deviation)

Measures how much the price is fluctuating recently

Helps the model understand risk and uncertainty in price movements




🎯 Prediction Target:
We use the current and past values of the above features to predict:

Target = BAC(t)  # The closing price of BAC for the next day
So, the model learns to map:

Input (t-1 values of BAC, JPM, macro indicators, technical indicators) → Output (BAC price at time t)





## 🤖 How is Machine Learning Used in Stock Prediction?


Machine Learning (ML) is used in stock prediction to discover patterns and relationships between various market indicators and future stock prices — something traditional models often fail to capture.

In this project, we treat it as a supervised regression problem where:

Inputs (X) = Past data of stocks, technical indicators, and macroeconomic indicators

Output (Y) = Next-day closing price of Bank of America (BAC)

🧠 Key Concepts Used:
✅ Regression Approach
We use regression models because stock price is a continuous numeric value (not category). ML models try to learn a function that maps input features to future price.

🔁 Lag Features Capture Time Dependency
Lag features like BAC(t-1), JPM(t-1) capture yesterday’s values, helping the model learn sequential patterns — essential for time series forecasting.

🌍 Context Through Multiple Features
Instead of using only BAC's own history, we also use:

Peer stock movements (e.g., JPM, MS)

Macroeconomic signals (e.g., VIX, TNX, Oil, Gold)

Technical indicators (e.g., Moving Average, Volatility)

This gives the model a broader context, improving its understanding of how BAC behaves under different market conditions.

📏 Evaluation Metrics
We evaluate our ML models using:

Metric	Description
R² Score	How much of the variation in actual prices is explained by the model (1.0 = perfect)
MSE	Mean of squared errors between actual and predicted values
RMSE	Square root of MSE; shows average prediction error in the same unit as stock price (dollars)

These metrics help identify which model performs best and how reliable the predictions are.





## 📊 Models Used in This Project

To predict the next-day closing price of BAC stock, we explored four supervised regression models. Each model applies a different strategy to identify underlying patterns in the data and generate price predictions.

1. Decision Tree Regressor
This model builds a tree-like structure of decisions based on feature thresholds. It recursively splits the dataset into smaller groups, where each decision node is based on minimizing prediction error.

🔍 Why it's useful:
Captures non-linear interactions between features.

Adapts well to datasets with mixed signals.

Can highlight which features are most important in the decision process.

⚙️ Key Behavior:
Performs well when few dominant features influence the target.

May become too specific if not constrained (like using max_depth).

2. Random Forest Regressor
Random Forest is an ensemble method that combines predictions from many decision trees. Each tree is trained on a random subset of data and features, promoting diversity.

🔍 Why it's useful:
Improves stability and accuracy over a single tree.

Naturally handles noise and overfitting.

Capable of learning from large feature sets with high correlation.

⚙️ Key Behavior:
Learns diverse perspectives of data and balances them.

Each individual tree may overfit, but the forest as a whole generalizes well.

3. K-Nearest Neighbors Regressor
KNN predicts the output by looking at the K most similar past observations. It uses distance metrics (usually Euclidean) to determine "nearness" between feature vectors.

🔍 Why it's useful:
Requires no training time.

Useful for detecting local structure in data.

⚙️ Key Behavior:
Sensitive to irrelevant features and feature scale.

May struggle with high-dimensional or sequential data.

4. Support Vector Regressor (SVR)
SVR attempts to fit a function within a defined margin (epsilon) around the true values. It supports non-linear modeling via kernel transformations.

🔍 Why it's useful:
Excellent for capturing complex non-linear relationships.

Robust to outliers within the epsilon margin.

⚙️ Key Behavior:
Requires careful tuning of parameters (C, epsilon, kernel).

May not generalize well on large or unscaled datasets without preprocessing.





## 📂 What Data Was Collected and Why?

To ensure the model could make informed predictions, we curated a diverse set of financial and economic indicators that influence BAC's stock price. The data collected spanned a long-term range to capture different market cycles and structural patterns.

📅 Date Range Covered
2002-01-01 to 2024-12-31

Over two decades of data to expose the model to:

Bull/bear markets

Financial crises (2008)

Post-COVID volatility

Rate hikes, inflation periods

🧾 Sources and Signals Used
We divided the data into logical categories based on the type of influence each feature might have on the stock price.

🔹 1. Primary Target
Ticker	Description
BAC	Bank of America – The stock we want to predict

The model is trained to forecast next-day closing price of this stock.

🔹 2. Sector Peers (Banks)
Tickers	Description
JPM, C, MS, WFC	Major US banks and financials

These are included to reflect co-movement within the banking sector. Financial stocks often respond similarly to economic news and investor sentiment.

🔹 3. Macroeconomic Factors
Tickers	Description
^TNX	10-Year Treasury Yield – rate impact
^VIX	Volatility Index – market fear gauge
CL=F	Crude Oil Futures – global cost signal
GC=F	Gold Futures – safe haven sentiment
DX-Y.NYB	US Dollar Index – currency strength

These were selected to account for external pressures affecting banking operations, credit demand, and market liquidity.

🔹 4. Overall Market Indicator
Ticker	Description
SPY	S&P 500 ETF – market-wide trend proxy

Including SPY helps the model understand general investor behavior and systemic moves in the broader US stock market.

🧠 Selection Criteria Summary
All features were chosen based on:

Relevance to financial markets

Coverage across economic and corporate signals

Availability and consistency over time

This comprehensive dataset ensures the model is aware of both micro and macro drivers of BAC’s price.





## 🧹 How Missing Values Were Handled

Handling missing data correctly is critical in time series forecasting, especially in finance where even small gaps can distort learning. In this project, we chose a method that respects the temporal structure of the dataset.

🛠️ Method Used:

df.fillna(method='ffill')


✅ Why This Approach?
Forward Fill (ffill) propagates the last known valid value forward until a new one is available.

💡 Reasons for Choosing Forward Fill:
Preserves Temporal Integrity

Maintains the sequence of values over time.

Ensures that future values aren't used to fill earlier timestamps (which would lead to data leakage).

Assumes Market Stability During Missing Data

In real-world trading, if a data point is missing (e.g., a stock didn’t trade that day), it’s reasonable to assume the last known value persisted.

Prevents Artificial Signals

Techniques like mean or median imputation can introduce unrealistic flat lines or shifts into time series, confusing the model.

Ensures Feature Alignment

Since all financial data was time-aligned, ffill guarantees that no row is partially filled or misaligned during model training.

❗ Why Not Mean, Median, or Interpolation?
Mean/Median would destroy price continuity and inject artificial behavior.

Interpolation could falsely simulate price action that never occurred — not safe for predictive models in finance.

This method gave us a clean, consistent, and temporally honest dataset — ready for feature engineering and modeling.




## 🛠️ Feature Engineering Techniques

Feature engineering is the heart of this project. It helps translate raw financial data into meaningful input signals that a machine learning model can understand. We focused on crafting features that reflect temporal structure, trends, and uncertainty in the market.

1. ⏮️ Lag Features
We created lag versions of each input feature using the .shift() function.


df['JPM(t-1)'] = df['JPM'].shift(1)
🔍 Purpose:

Allows the model to "see" what happened yesterday.

Captures immediate market memory.

Especially important for stocks with autocorrelation (where today’s price depends on yesterday).

2. 📊 Moving Averages
Moving averages provide a smoothed version of the price, ignoring short-term noise.

df['BAC(ma_5)'] = df['BAC'].rolling(5).mean()
🔍 Purpose:

Helps model detect short-term price trends.

Widely used by traders to identify momentum.

Acts as a simple trend indicator in the feature set.

3. 🌪️ Rolling Volatility
Volatility is calculated using rolling standard deviation.


df['BAC(vol_5)'] = df['BAC'].rolling(5).std()
🔍 Purpose:

Measures risk or unpredictability in price movement.

Helps the model respond to changing market conditions (stable vs volatile).

Valuable in finance where price behavior during high volatility can differ significantly.

4. 🎯 Target Variable
The prediction target is the next-day closing price of BAC.

df['target'] = df['BAC'].shift(-1)
🔍 Purpose:

Converts the task into a supervised regression problem.

Ensures the model is trained to forecast the future, not the current day.

🚿 Post Feature-Creation Cleanup
Once all features were engineered, we used:

df.dropna(inplace=True)

🔍 Why?

Rolling windows and shifts naturally introduce NaNs.

Removing incomplete rows ensures consistency across all input features during training.

These engineered features gave our models the ability to recognize direction, strength, and volatility of price movements, making the predictions smarter and more responsive to market conditions.



## 🔀 Train-Test Split

In time series forecasting, it's essential to preserve the chronological order of data when training machine learning models. Unlike traditional datasets where shuffling is fine, stock market data is sequential — meaning future prices must not influence past observations during training.

🧪 Method Used
We used scikit-learn’s train_test_split() function with:

shuffle=False
This ensures that the most recent (future) data is reserved for testing, and only past (historical) data is used for training.

🔎 Why This Matters
📅 Respects Temporal Order
The goal is to simulate how a real-world model would work — trained on past data and tested on future prices.

🔐 Prevents Data Leakage
Shuffling data would mix future events into training, making the model unrealistically accurate.

⚖️ Realistic Backtesting
Financial models must perform well on unseen future data, just like in actual trading environments.

📊 Split Ratio
We used an 80:20 split:

80% of data (from 2002 onward) → used to train the model

20% (most recent years) → used to evaluate prediction performance

This ensures the model is validated on the latest market behavior, making evaluation more reliable.



## 📈 Model Evaluation and Results
 
After training the models, we evaluated their performance on the test set using three key regression metrics:

Metric	Description
R² Score	Measures how well the model explains variability in the target. Closer to 1 means better fit.
MSE (Mean Squared Error)	Average of squared prediction errors. Lower is better.
RMSE (Root Mean Squared Error)	Square root of MSE. Interpretable in the same unit as the stock price (dollars).

📊 Summary of Results:

| Model         | R² Score | MSE   | RMSE |
| ------------- | -------- | ----- | ---- |
| Decision Tree | 0.96     | 1.21  | 1.10 |
| Random Forest | 0.98 ✅   | 0.52  | 0.72 |
| KNN           | -0.59 ❌  | 51.02 | 7.14 |
| SVR           | 0.01 ❌   | 31.67 | 5.63 |


✅ Best Performing Model: Random Forest Regressor

It achieved the lowest RMSE and highest R², indicating strong predictive accuracy on unseen data.




## 🥇 Why the Best Model Performed Best


The Random Forest Regressor delivered superior results due to several key strengths that aligned well with our dataset:

🔍 1. Handles Complex Relationships
Random Forest models can capture non-linear interactions among variables — which is common in financial data.

🧱 2. Ensemble Strategy Reduces Overfitting
By combining predictions from many decision trees, it averages out noise and avoids over-relying on specific patterns.

📈 3. Scales Well with High Feature Count
With lag features, macro indicators, and technical signals, our input space was large. Random Forest handles such high-dimensional feature sets effectively.

🧪 4. Robust to Noisy or Weak Features
Not every input contributes equally. Random Forest can ignore less informative variables internally, improving generalization.

Together, these strengths made it an ideal choice for financial time series prediction — balancing accuracy, robustness, and reliability.



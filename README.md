# GlucosePrediction

This project is a machine learning-based glucose level predictor that estimates future glucose concentrations based on historical data. It leverages time series modeling techniques to analyze trends and fluctuations in glucose readings over time. The goal is to provide users—especially those managing diabetes or glucose-related health conditions—with accurate and timely predictions to support better decision-making and health monitoring. I experimented with three different predictive models to assess their effectiveness:
1) Linear Regression: This served as a baseline model, achieving an accuracy of approximately 82%. While simple and interpretable, it struggled to capture the temporal dependencies inherent in glucose dynamics.
2) LSTM (Long Short-Term Memory) Network: Leveraging the sequential nature of time-series data, the LSTM model improved performance to 84% accuracy. This recurrent neural network was better suited for learning temporal trends in the glucose readings.
3) XGBoost (Extreme Gradient Boosting): This model significantly outperformed the others, achieving 97% accuracy. XGBoost’s ability to handle nonlinear relationships and its robustness against overfitting made it a powerful choice for this task. The graph illustrating its performance is attached below for reference.


<img width="1134" alt="image" src="https://github.com/user-attachments/assets/f594e954-b822-4b45-ba08-8bb0733119ab" />

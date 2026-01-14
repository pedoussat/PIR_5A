# 5A-Research_Project: Transilien Passenger Affluence Prediction
This project implements LSTM deep learning models to predict passenger traffic across the Transilien network.
https://challengedata.ens.fr/participants/challenges/149/

## Data
The file containing the x_train data is too large, so please find all the data on this drive.

https://drive.google.com/drive/u/0/folders/1EW-k1jYezIveOdagwA_kmZXe0bI-L3Uk


## Project Architecture
See the ProjectArchitecture.pdf file.

## Submission History (Attempts)
| File | Type  | Description | Hyperparameter Optimization | Features | Score |
| :--- | :---  |     :--     |   :---                      |  :---    |  :--- |
| **y_test_LSTM_v3.1_sorted** | Non Autoregressive model | LSTMn°3, no validation data | No  | \['job', 'ferie', 'vacances'\]      |  **219.14**  |
| **y_test_LSTM_v3.2_sorted** | Non Autoregressive model | LSTMn°3, no validation data | Yes | \['job', 'ferie', 'vacances'\]      |  **209.61**  |
|  **y_test_LSTM_v4_sorted**  | Non Autoregressive model | LSTMn°4, validation data    | Yes | \['job', 'ferie', 'vacances'\]      |  **249.12**  |

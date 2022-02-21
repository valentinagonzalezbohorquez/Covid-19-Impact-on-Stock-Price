library(quantmod)
library(forecast)
library(tseries)
library(timeSeries)
library(dplyr)
library(readxl)
library(kableExtra)
library(data.table)
library(DT)
library(tsfknn)
library(Metrics)

#### Importing the data

getSymbols("MSFT", src = "yahoo", from = "2015-01-01", to = "2020-02-28")
microsoft_data_before_covid <- as.data.frame(MSFT)
tsData_before_covid <- ts(microsoft_data_before_covid$MSFT.Close)


getSymbols("MSFT", src = "yahoo", from = "2020-03-02")
microsoft_data_after_covid <- as.data.frame(MSFT)
microsoft_data_after_covid


#### Graphical representation of the Data
par(mfrow = c(1,1))
plot.ts(tsData_before_covid, ylab = "Closing Price", main = "Before COVID-19")

#### Dataset Preview
datatable(microsoft_data_before_covid, filter = 'top')
#### ARIMA Model
##Let us first analyse the ACF and PACF Graph of each of the two datasets.
par(mfrow = c(2,1))
acf(tsData_before_covid, main = "Before COVID-19")
pacf(tsData_before_covid, main = "Before COVID-19")

## Conduct an ADF (Augmented Dickey-Fuller) test 
## for the stationarity of the time series data for both the datasets closing price.

print(adf.test(tsData_before_covid))

#p-value greater than 0.05 indicates that the time series data is non-stationary

## KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test to check
## for the stationarity of the time series data for both the datasets closing price.
print(kpss.test(tsData_before_covid))

#p-value less than 0.05 for this test indicates timeseries data is NOT stationary

#Since both tests indicate non-stationarity, we'll assume that the timeseries data is not stationary. 

#### Use the auto.arima function to determine the time series model for each of the datasets.
modelfit_before_covid <- auto.arima(tsData_before_covid, lambda = "auto")
summary(modelfit_before_covid)

arimaorder(modelfit_before_covid)

#ARIMA(3,1,3) has been suggested as a good fit by the auto.arima() function


## Perform residual diagnostics for each of the fitted models
par(mfrow = c(1,3))

plot(modelfit_before_covid$residuals, ylab = 'Residuals', main = "Before COVID-19")
acf(modelfit_before_covid$residuals,ylim = c(-1,1), main = "Before COVID-19")
pacf(modelfit_before_covid$residuals,ylim = c(-1,1), main = "Before COVID-19")

#From the residual plot , we can confirm that the residual has a mean of 0 and the variance is constant as well . The ACF is 0 for lag> 0 , and the PACF is 0 as well.So, we can say that the residual behaves like white noise and conclude that the model ARIMA(3,1,3) fits the data well.

#### Using the Box-Ljung Test test at a significance level of 0.05 if residual follow white noise.
Box.test(modelfit_before_covid$residuals, type = "Ljung-Box")

## p-value greater than 0.05; so, the Box-Ljung test confirms that the residuals follow a white noise. 

#### KNN Regression Time Series Forecasting Model
par(mfrow = c(1,1))

#The general rule of thumb for selecting the value of k is taking the square root of the number of data points in the sample.
kvalue = as.integer(sqrt(length(microsoft_data_before_covid$MSFT.Close)))
hvalue = 30
predknn_before_covid <- knn_forecasting(microsoft_data_before_covid$MSFT.Close, h = hvalue, lags = 1:30, k = kvalue, msas = "MIMO")

plot(predknn_before_covid, main = "Predictions based on before Covid data")


#### Evaluating the KNN model for our forecasting time series; using rolling_origin() to obtain the forecast errors at several points
knn_ro_before_covid <- rolling_origin(predknn_before_covid)

#Taking the global RMSE, MAE, MAPE
knn_ro_before_covid$global_accu

#### Feed Forward Neural Network Modelling
#Hidden layers creation
alpha <- 1.5^(-10)
hn_before_covid <- length(microsoft_data_before_covid$MSFT.Close)/(alpha*(length(microsoft_data_before_covid$MSFT.Close) + hvalue))

#Fitting nnetar
lambda_before_covid <- BoxCox.lambda(microsoft_data_before_covid$MSFT.Close)
dnn_pred_before_covid <- nnetar(microsoft_data_before_covid$MSFT.Close, size = hn_before_covid, lambda = lambda_before_covid)

# Forecasting Using nnetar
dnn_forecast_before_covid <- forecast(dnn_pred_before_covid, h = hvalue, PI = TRUE)

plot(dnn_forecast_before_covid, main = "Predictions based on Neural Network Model")


#### Analyze the performance of the neural network model using the following parameters
forecast::accuracy(dnn_forecast_before_covid)

summary_table_before_covid <- data.frame(Model = character(), RMSE = numeric(), MAE = numeric(), 
                                         MAPE = numeric(), stringsAsFactors = FALSE)

summary_table_before_covid[1,] <- list("ARIMA", summary(modelfit_before_covid)[2], summary(modelfit_before_covid)[3], summary(modelfit_before_covid)[5])
summary_table_before_covid[2,] <- list("KNN", knn_ro_before_covid$global_accu[1], knn_ro_before_covid$global_accu[2], knn_ro_before_covid$global_accu[3])
summary_table_before_covid[3,] <- list("Neural Network", forecast::accuracy(dnn_forecast_before_covid)[1,2], forecast::accuracy(dnn_forecast_before_covid)[1,3], forecast::accuracy(dnn_forecast_before_covid)[1,5])



kable(summary_table_before_covid, caption = "Summary of Models for data before COVID-19") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), full_width = F, fixed_thead = T )


#ARIMA and Neural Network models performed quite similarly. We'll use Neural network model to get the predictions and compare with actual values for a range of 30 days starting from 2nd March 2020.

forecast_during_covid <- data.frame("Date" = row.names(head(microsoft_data_after_covid, n = 30)),
                                    "Actual Values" = head(microsoft_data_after_covid$MSFT.Close, n = 30),
                                    "Forecasted Values" = dnn_forecast_before_covid$mean)


datatable(forecast_during_covid, filter = 'top')

outputtable <- as.data.frame(forecast_during_covid)
outputtable
#Calculating the RMSE based on the model predictions and Actual values. 
rootmeansquareerror = rmse(outputtable$Actual.Values, outputtable$Forecasted.Values)

rootmeansquareerror



#Observation: MSFT has performed better than predictions for the initial few days but later performed much worse than the predictions than the remaining time. 
#Reasoning: The model is built on pre-covid data. So, it's predictions do not include the impact due to Covid situation
#Conclusion: MSFT was likely impacted negatively due to Covid. 

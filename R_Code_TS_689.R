#install.packages('xts')
library(ggplot2)
library(xts)
library(forecast)
library(tseries)
library(fUnitRoots)

## Reading the data & converting it into timeseries

snp = read.csv("HistoricalPrices.csv")
snp$Date = as.Date(snp$Date)

snp_ts = xts(snp[,-1],order.by = snp$Date)

snp_weekly=to.weekly(snp_ts)
snp_qtrly = to.quarterly(snp_ts)

## Reading GDP data (Optional)

gdp = read.csv("GDP Historical Data.csv")
gdp$Date = as.Date(gdp$Date)
gdp

gdp_ts = xts(gdp[,-1],order.by = gdp$Date)
gdp_ts

gdp_qtrly = to.quarterly(gdp_ts)
gdp_qtrly = gdp_qtrly[,1]
colnames(gdp_qtrly) = 'Value'

## Calculating growth rate

growth_rate = 100*diff(log(gdp_qtrly))

## Augmented Dickey Fuller Test on S&P 500

adf.test(snp_qtrly$snp_ts.Close, alternative = c("stationary"), k=6)
# shows that the series has a stochastic trend

## Taking 1st order difference
dclose=diff(log(snp_qtrly$snp_ts.Close))[-1,]

adf.test(dclose, alternative = c("stationary"), k=6)
# now it is stationary

## Creating ARIMA model and taking recursive forecast
t=74
T=length(snp_qtrly$snp_ts.Close)
snp_predict_ar=matrix(NA,t,1)
for(i in t:0){
  snp_temp=snp_qtrly$snp_ts.Close[1:(T-i)]
  m_predict=auto.arima(snp_temp, max.p = 12, max.q = 10, d = 1,
                       seasonal = FALSE, ic='bic')
  snp_predict_ar[t-i+1]=forecast(m_predict,h=1)$mean
}

snp_close=ts(snp_qtrly$snp_ts.Close[21:94],start=c(2000,01),frequency=4)
snp_predict_ar=ts(snp_predict_ar,start=c(2000,01),frequency=4)
data_combine=cbind(snp_close,snp_predict_ar)
plot(data_combine,main="Real vs Predicted S&P 500",plot.type=c("single"),xlab = "Date",ylab="Stock Price",col=c("blue","red"),lty=1:2)
legend("bottomright",legend=c("Data","Forecast"),col=c("blue","red"),lty=1:2)

MSE_snp= mean((data_combine[1:74,1]- data_combine[1:74,2])^2)

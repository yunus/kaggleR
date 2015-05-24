# This script has been inspired by the  Random Forests benchmark code.
# from kaggle, bike challenge (https://www.kaggle.com/c/bike-sharing-demand).
#
# You can find the full dataset from http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
# In the full data set, different from the bike challenge, there are additional features and all the features have already been scaled.
# So you need to change the code for the full data set. 

library(lubridate)
library(caret)
library(e1071)
library(doMC)
library(Metrics)
registerDoMC(cores=3)

set.seed(7)


training <- read.csv("train.csv",stringsAsFactors=FALSE)
test <- read.csv("test.csv",stringsAsFactors=FALSE)

# for local testing I have created a cv set though Caret does k-fold cv for parameter tunning
# for final submission I use whole training set.
cv <- training[mday(ymd_hms(training$datetime)) > 13,]
training <- training[mday(ymd_hms(training$datetime)) <= 13,]



normalizationParameters <- data.frame()

meanNormalize <- function(data,columns) {
  normalizationParameters 
  for(col in columns){
    if(!(col %in% colnames(data) )){next}
    temp = data[,col]
    if( !(col %in% rownames(normalizationParameters) ) ){
      # we have fixed mean and standard deviation for the training set, 
      # for testing skip
      normalizationParameters[col,"m"] <-  mean(temp)
      normalizationParameters[col,"r"] <-  sd(temp) 
    }
    if(col == "count" | col=="casual" | col=="registered"){
      data[,col] <- log(temp+1) 
    }else {
      data[,col] <- (temp-normalizationParameters[col,"m"])/normalizationParameters[col,"r"]
    }
  }
  return(list(data,normalizationParameters))
}

denormalize <- function(vector,col){
  if(col=="count" | col=="casual" | col=="registered"){
    return( exp(vector) -1)
  } else {
    return( vector*normalizationParameters[col,"r"] + normalizationParameters[col,"m"])
  }
}


extractFeatures <- function(data) {
  features <- c("season",
                "holiday",
                "workingday",
                "weather",
                "temp",
                "atemp",
                "humidity",
                "windspeed", "h","wday","year","mth","week","mday","datetime")
  if("count" %in% colnames(data) ){features <- append(features, "count")}
  if("casual" %in% colnames(data) ){features <- append(features, "count")}
  if("registered" %in% colnames(data) ){features <- append(features, "count")}
  data$date <- ymd_hms(data$datetime)
  data$h <- hour(data$date)
  data$wday <- wday(data$date)
  data$year <- year(data$date)
  data$mday <- mday(data$date)
  # the test set includes the days of month larger than 20. I wanted to put the estimated dates in the middle of a month instead of the end.
  # therefore, I created a new month feature.
  data$mth <- ifelse( data$mday > 10, month(ymd_hms(data$datetime)), month(ymd_hms(data$datetime)) -1 )
  data$week <- week(ymd_hms(data$datetime))
  return(data)
}

norm.columns <- c("temp","atemp","humidity","windspeed")

normalizationParameters <- as.data.frame(meanNormalize(rbind(training[,norm.columns],test[,norm.columns]), norm.columns)[2])

norm.columns <- append(norm.columns,c("count","mth","hour","casual","registered"))

# normalize and extract features
temp <- meanNormalize( extractFeatures(training), norm.columns )
trainFea <- as.data.frame(temp[1])
normalizationParameters <- as.data.frame(temp[2])

temp <- meanNormalize( extractFeatures(test), norm.columns )
testFea  <- as.data.frame(temp[1])
temp <- meanNormalize(extractFeatures(cv), norm.columns )
cvFea  <- as.data.frame(temp[1])
rm(temp)

submission <- data.frame(datetime=test$datetime, count=NA,casual=NA,registered=NA)
submissiontrain <- data.frame(datetime=training$datetime, count=NA,casual=NA,registered=NA)
submissioncv <- data.frame(datetime=cv$datetime, count=NA,casual=NA,registered=NA)


svmtune.grid <- expand.grid(sigma=2^(-5:2),C=2^(-3:3))
trControl <- trainControl(method = "repeatedcv", number = 10)


# parameters for the RFE function
#rfeControl <- rfeControl(functions = caretFuncs, method = "boot",
 #                        number= 10, verbose = FALSE,repeats=3 )

# the most important feature is hour, then day of week. We let the machine learning algorithm deal with hour, but separated the days. 
# The reason is that, in weekends the distribution changes totally.
for (i_wd in unique(trainFea$workingday)) { 
  for (i_year in unique(trainFea$year)){
    
    cat( " workingday",i_wd," year",i_year,"\n")
    #TEST
    testLocs   <-    testFea$workingday == i_wd  & testFea$year==i_year #
    testSubset <- testFea[testLocs,]
    #CV
    cvLocs   <-    cvFea$workingday == i_wd & cvFea$year==i_year #
    cvSubset <- cvFea[cvLocs,]
    #TRAIN
    trainLocs  <-   trainFea$workingday == i_wd &  trainFea$year == i_year #
    trainSubset <- trainFea[trainLocs,]
    
    
    
    
    fit.casual <- train(casual ~ wday+weather+windspeed+temp+humidity+holiday+mth+h,
                 data = trainSubset, method="svmRadial",
               #  rfeControl = rfeControl, 
                 trControl = trControl, 
                 tuneGrid=svmtune.grid, scale=FALSE)
    fit.registered <- train(registered ~ wday+weather+windspeed+temp+humidity+holiday+mth+h,
                        data = trainSubset, method="svmRadial",
              #          rfeControl = rfeControl, 
                        trControl = trControl, 
                        tuneGrid=svmtune.grid, scale=FALSE)
    
 
    
    submission[testLocs, "casual"] <- predict(fit.casual, testSubset)
    submissioncv[cvLocs, "casual"] <- predict(fit.casual, cvSubset)
    submissiontrain[trainLocs,"casual"] <- predict(fit.casual,trainFea[trainLocs,])
    submission[testLocs, "registered"] <- predict(fit.registered, testSubset)
    submissioncv[cvLocs, "registered"] <- predict(fit.registered, cvSubset)
    submissiontrain[trainLocs,"registered"] <- predict(fit.registered,trainFea[trainLocs,])


    
  } # year
} #  workingday



cat("train RMSLE registered and casual error is ",rmsle(training$count, denormalize(submissiontrain$casual,"casual") +
                                                                            denormalize(submissiontrain$registered,"registered")),"\n")
cat("cv RMSLE  registered and casual  error is ",rmsle(cv$count, denormalize(submissioncv$casual,"casual") +
                                                          denormalize(submissioncv$registered,"registered")),"\n")

submission$count <- round( denormalize(submission$casual,"casual") + 
                            denormalize(submission$registered,"registered"))
write.csv(subset(submission,select=c("datetime","count")), 
          file = "submission.csv", row.names=FALSE)



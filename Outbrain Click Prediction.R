# Project- Click Predictions
# Team Power Rangers
# Members- Dheeraj Srivathsav and Vimlesh Bavadiya



library(data.table) # to load the data faster
library(ggplot2)      # For advanced visualizations
library(Hmisc)        # For calculating of correlation test
library(PerformanceAnalytics) # For creating histogram and correlation coeff matrix
library(EnvStats)     # For boxcox transformations
library(ggbiplot)    # For visualizing principal components
library(MASS)         # For performing LDA 
library(Amelia)       #  To access Freetrade Data
library(ForImp)      # For Imputation
library(mice)         # To access imputation functions
library(randomForest) # To access random forest imputation
library (lsr)       # To calculate mean absolute deviation
library(stargazer)  # To create output tables
library(caret)        # For classification and regression training
library(fBasics)    # For descriptive statistics
library(outliers)   # For outlier functions
library (VIM)       # for calculating missing values
library(pls)        #load the plsr package for Partial Least Squares Regression
library(car)        #load car to get vif
library(lars)         # for lasso
library(glmnet)     # For LASSO and elasticnet
library(elasticnet)   # for elastic net regression
library(pls)         # lbrary for pls regression
library(gmodels)    # for cross-table function similar to SAS
library(InformationValue)  # For model evaluation function
library(MLmetrics)  # for Logloss function
library(PerformanceAnalytics) # for performance functions
library(rpart)         # CART algorithm
library(party)         # to print trees
library(partykit)      # to print trees using "party"
library(rattle)        # for graphics
library(adabag)        # for boosting
library(ipred)         # for bagging and error estimation
library(randomForest)  # for Random Forests
library(caret)         # for training and modeling
library(MASS)          # to obtain the sample data set "Glass"
library (ROCR)        #  to get ROC curve
library(rfUtilities)  # for random forest cross validation
library(AppliedPredictiveModeling)   # for predictive modelling using caret
library(caTools)      # for splitting data
library(devtools)     # for PCA analysis
library(robustbase)   #adjusted boxplots
library(mboost)       # fo glm boost
library(gbm)        #gbm
library(readr)
library(dplyr)
library(googleVis)    # for world map data
library(dummies)      # for one hot encoding


Model.eval=function(truevalue,fittedvalue,predictedvalue){
  
  #---------------   Confusion Matrix and Statistics   ---------------------------------  
  Confusionmatrix=caret::confusionMatrix(predictedvalue, truevalue, positive="1")
  
  #---------------   ROC Curve and AUC   --------------------------------- 
  pred <- prediction(fittedvalue, truevalue)    
  perf <- performance(pred,"tpr","fpr")
  pred1=prediction (truevalue,truevalue)
  plot(perf,colorize=TRUE, main="ROC curve",print.cutoffs.at = c(0.25,0.5,0.75)); 
  abline(0, 1, col="red") 
  
  auc <- performance(pred, measure = "auc")
  auc1 <- auc@y.values[[1]]
  
  #---------------   optimizing cutoff based on cost   ---------------------------------   
  cost.perf = performance(pred, "cost")
  Optcutoff=pred@cutoffs[[1]][which.min(cost.perf@y.values[[1]])]
  
  #---------------   Concordant pairs   --------------------------------- 
  Conpairs=Concordance(truevalue, predictedvalue)
  
  #---------------   D-statistics   ---------------------------------   
  Dstats<-mean(fittedvalue[truevalue==1]) - mean(fittedvalue[truevalue==0])
  
  #---------------   K-S chart and statistic   --------------------------------- 
  Ksstats=ks_stat(truevalue, predictedvalue)
  ks_plot(truevalue, predictedvalue)
  
  #---------------   Distribution of predicted probabilities values for true positives and true negatives   ---------------------------------  
  plot(0,0,type="n", xlim= c(0,1), ylim=c(0,7),     
       xlab="Prediction", ylab="Density",  
       main="Distribution of predicted probability for true positives and ture negatives")
  
  for (runi in 1:length(pred@predictions)) {
    lines(density(pred@predictions[[runi]][pred@labels[[runi]]==1]), col= "blue")
    lines(density(pred@predictions[[runi]][pred@labels[[runi]]==0]), col="green")
  } 
  
  #---------------   Cumulative Gains Chart   --------------------------------- 
  
  gain <- performance(pred,"tpr","rpp")
  gain1= performance(pred1,"tpr","rpp")
  
  plot(x=c(0, 1), y=c(0, 1), type="l", col="red", lwd=2,
       main="Cumulative Gains Chart",
       ylab="True Positive Rate", 
       xlab="Rate of Positive Predictions")
  
  
  gain.x = unlist(slot(gain, 'x.values'))
  gain.y = unlist(slot(gain, 'y.values'))
  
  lines(x=gain.x, y=gain.y, col="orange", lwd=2)
  
  gain1.x = unlist(slot(gain1, 'x.values'))
  gain1.y = unlist(slot(gain1, 'y.values'))
  
  lines(x=gain1.x, y=gain1.y, col="darkgreen", lwd=2)
  
  #---------------   Log Loss   --------------------------------- 
  
  Logloss1=LogLoss(fittedvalue, truevalue)
  
  
  return(list("Confusion Matrix and Statistics"=Confusionmatrix,
              "AUC"=auc1,
              "Optimum Threshhold"=Optcutoff,
              "Concordance Pairs"= Conpairs,
              "D Statistics"=Dstats,
              "K-S Statistics"=Ksstats,
              "Logloss"=Logloss1))
}




##################################       Reading the data files into data frames    #######################################################
# Setting up working directory
getwd()
setwd("C:/Users/Drilling Simulator/Documents/Project/Data")

# Reading data and storing them in dataframe
events<- fread('events.csv')
#clicks_page_views <- fread('page_views_sample.csv') # not using page views for training
clicks.train<- fread('clicks_train.csv')
#clicks_test<- fread('clicks_test.csv')              # since we are not submitting to the competition we are not using test data
prom.cont<- fread('promoted_content.csv')
doc.meta<- fread('documents_meta.csv')
doc.cate= fread('documents_categories.csv')
doc.enti= fread('documents_entities.csv')
doc.topics= fread('documents_topics.csv')

doc.cate= fread('D:/OU_SEMISTER_1/IDA/Project/Data set/documents_categories.csv')
doc.cate.max.conf=fread('D:/OU_SEMISTER_1/IDA/Project/sampled data/doc.cate.max.conf.csv')
doc.topics=fread('D:/OU_SEMISTER_1/IDA/Project/Data set/documents_topics.csv')

##################################        Data Exploration           ######################################################################

# looking at structure and summary of events
head(events)
tail(events)
str(events)
summary(events)

# looking at structure and summary of train
head(clicks.train)
tail(clicks.train)
str(clicks.train)
summary(clicks.train)

# looking at structure and summary of promoted content
head(prom.cont)
tail(prom.cont)
str(prom.cont)
summary(prom.cont)

# looking at structure and summary of documents meta
head(doc.meta)
tail(doc.meta)
str(doc.meta)
summary(doc.meta)

# looking at structure and summary of documents category
head(doc.cate)
tail(doc.cate)
str(doc.cate)
summary(doc.cate)

# looking at structure and summary of document entities
head(doc.enti)
tail(doc.enti)
str(doc.enti)
summary(doc.enti)

# looking at structure and summary of document topics
head(doc.topics)
tail(doc.topics)
str(doc.topics)
summary(doc.topics)

# Exploring missingness of data
md.pattern(events)
md.pairs(events)
aggr(events)
missmap(events,y.cex=.5, x.cex=.6)
misdata=aggr(events)
misdata

# It took long time to find missing data hence summary was used for quick analysis

# Exploring the train dataset
clicks.trainad<- clicks.train[ , .N , keyby="ad_id"]
clicks.traindis<- clicks.train[ , .N , keyby="display_id"]
head(clicks.trainad)
summary(clicks.trainad)
head(clicks.traindis)
summary(clicks.traindis)

# Plotting histogram of no. of times ad appeared in train data
ggplot(data=clicks.trainad, aes(clicks.trainad$N)) + 
  geom_histogram(breaks=seq(0, 250000, by = 5000), 
                 col="blue", 
                 fill="blue", 
                 alpha = .2) + 
  labs(title="Histogram of no. of times ad appeared") +
  labs(x="No. of times ad appeared", y="Frequency") + 
  xlim(c(0,250000)) + 
  scale_y_continuous(trans="log10")
  #scale_y_log10(breaks=c(0,10000))
  #ylim(c(0,1000000))

# Plotting histogram of no. of ad_id in display_id in train data
ggplot(data=clicks.traindis, aes(clicks.traindis$N)) + 
  geom_histogram(breaks=seq(0, 15, by = 1), 
                 col="blue", 
                 fill="blue", 
                 alpha = .2) + 
  labs(title="Histogram for no. of ad_id in display_id") +
  labs(x="No of ad_id in a display_id", y="Frequency") + 
  xlim(c(0,15)) + 
  scale_y_continuous(trans="log10")

# Exploring data in Events
events.platform<- events[ , .N , keyby="platform"]
events.disid<- events[ , .N , keyby="display_id"]
events.uuid<- events[ , .N , keyby="uuid"]
events.docid<- events[ , .N , keyby="document_id"]
events.geoloc<- events[ , .N , keyby="geo_location"]

View(events.platform)
summary(events.platform)
summary(events.disid)
summary(events.uuid)
summary(events.docid)
summary(events.geoloc)

# Plotting histogrm for platform in events
#events.platform=events.platform[1:3,]
#events.platform
#str(events.platform)
#ggplot(data=events.platform, aes(events.platform$platform)) + 
#  geom_histogram(breaks=seq(0, 3, by = 1),
#                  col="blue", 
#                 fill="blue", 
#                 alpha = .2) + 
#  labs(title="Histogram for platform in events") +
#  labs(x="No of Platform", y="Frequency") + 
#  xlim(c(0,3)) + 
#  scale_y_continuous(trans="log10")

#hist(events.platform$platform, main="Bill5")

# Plotting histogram for user id
ggplot(data=events.uuid, aes(events.uuid$N)) + 
  geom_histogram(breaks=seq(0, 50, by = 1), 
                 col="blue", 
                 fill="blue", 
                 alpha = .2) + 
  labs(title="Histogram for no. of times uuid is repeated") +
  labs(x="No. of times uuid is repeated", y="Frequency") + 
  xlim(c(0,50)) + 
  scale_y_continuous(trans="log10")

# Plotting histogram for document id
ggplot(data=events.docid, aes(events.docid$N)) + 
  geom_histogram(breaks=seq(0, 325000, by = 5000), 
                 col="blue", 
                 fill="blue", 
                 alpha = .2) + 
  labs(title="Histogram for no. of times document_id is repeated") +
  labs(x="No. of times document_id is repeated", y="Frequency") + 
  xlim(c(0,325000)) + 
  scale_y_continuous(trans="log10")

# Plotting histogram for geo_location
ggplot(data=events.geoloc, aes(events.geoloc$N)) + 
  geom_histogram(breaks=seq(0, 1020000, by = 10000), 
                 col="blue", 
                 fill="blue", 
                 alpha = .2) + 
  labs(title="Histogram for no. of times geo_location is repeated") +
  labs(x="No. of times geo_location is repeated", y="Frequency") + 
  xlim(c(0,1020000)) + 
  scale_y_continuous(trans="log10")

# To find percentage of frequency of the data

#for(i in c(2,10,50,100,1000)){
#  for(j in 1:i-1){
#    value=count(clicks.train.ad, clicks.train.ad$N==j)+value
#  percent=(value/nrow(clicks.train.ad))*100
#    }
#  value
#  percent
#  }


#  counting the number of similar counts

s<-summary (as.factor(clicks.train.ad$N))
s
View(s)
value=0
length(which(clicks.trainad$N=="1"))

for(j in c(1:100)){
  value=length(which(clicks.trainad$N==j))+value
  percent=(value/nrow(clicks.trainad))*100
}
value
percent

# To get Worldmap and location data exploration

events$Country <- sapply(events$geo_location, function(x) strsplit(x, '>')[[1]][1])
events$State <- sapply(events$geo_location, function(x) strsplit(x, '>')[[1]][2])
events$DMA <- sapply(events$geo_location, function(x) strsplit(x, '>')[[1]][3])
head(events)

events$DMA2[is.na(events$DMA) & !is.na(as.numeric(events$State))] <- events$State[is.na(events$DMA) & !is.na(as.numeric(events$State))]
events$State[is.na(events$DMA) & !is.na(as.numeric(events$State))] <- NA
events$DMA[!is.na(events$DMA2)] <- events$DMA2[!is.na(events$DMA2)]
events$DMA2 <- NULL

event_summary <- events %>% 
  group_by(Country) %>% 
  summarise(count = n())
event_summary[order(-event_summary$count),]

event_summary.US <- events %>%
  filter(Country == 'US') %>%
  group_by(State) %>% 
  summarise(count = n())

event_summary.US_DMA <- events %>%
  filter(Country == 'US') %>%
  group_by(DMA) %>% 
  summarise(count = n())

event_summary.CA <- events %>%
  filter(Country == 'CA') %>%
  group_by(State) %>% 
  summarise(count = n())

View(event_summary)
write.csv(event_summary,"event_summary.csv")

event_summary <- fread('event_summary.csv')

# Plotting country data on world map
my_geo <- gvisGeoChart(event_summary %>% filter(Country != ''), "Country", "count",
                       options=list(width=600, height=400))
plot(my_geo)

# Plotting US states data on Map of USA 
my_geo2 <- gvisGeoChart(event_summary.US %>% filter(State != ''), "State", "count",
                        options=list(region="US",
                                     displayMode="regions", 
                                     resolution="provinces",
                                     width=600, height=400))
plot(my_geo2)

# Plotting DMA data om Map of USA
my_geo3 <- gvisGeoChart(event_summary.US_DMA %>% filter(DMA != ''), "DMA", "count",
                        options=list(region="US",
                                     displayMode="Marker",
                                     resolution="metros",
                                     width=600, height=400))
plot(my_geo3)

cat(my_geo$html$chart, file="count_by_country.html")
cat(my_geo2$html$chart, file="count_by_US_state.html")
cat(my_geo3$html$chart, file="count_by_US_DMA.html")

# Removing dataframes to free up the space
rm(clicks.trainad)
rm(clicks.traindis)
rm(event_summary)
rm(event_summary.CA)
rm(event_summary.US)
rm(event_summary.US_DMA)
rm(events.disid)
rm(events.docid)
rm(events.geoloc)
rm(events.platform)
rm(events.uuid)
rm(my_geo)
rm(my_geo2)
rm(my_geo3)
gc()

# setting the events back to the original dataset
events<- fread('events.csv')


#############################################       Function for model evaluations    #############################################
Model.eval=function(truevalue,fittedvalue,predictedvalue){
  
  #---------------   Confusion Matrix and Statistics   ---------------------------------  
  Confusionmatrix=caret::confusionMatrix(predictedvalue, truevalue, positive="1")
  
  #---------------   ROC Curve and AUC   --------------------------------- 
  pred <- prediction(fittedvalue, truevalue)    
  perf <- performance(pred,"tpr","fpr")
  pred1=prediction (truevalue,truevalue)
  plot(perf,colorize=TRUE, main="ROC curve",print.cutoffs.at = c(0.25,0.5,0.75)); 
  abline(0, 1, col="red") 
  
  auc <- performance(pred, measure = "auc")
  auc1 <- auc@y.values[[1]]
  
  #---------------   optimizing cutoff based on cost   ---------------------------------   
  cost.perf = performance(pred, "cost")
  Optcutoff=pred@cutoffs[[1]][which.min(cost.perf@y.values[[1]])]
  
  #---------------   Concordant pairs   --------------------------------- 
  Conpairs=Concordance(truevalue, predictedvalue)
  
  #---------------   D-statistics   ---------------------------------   
  Dstats<-mean(fittedvalue[truevalue==1]) - mean(fittedvalue[truevalue==0])
  
  #---------------   K-S chart and statistic   --------------------------------- 
  Ksstats=ks_stat(truevalue, predictedvalue)
  ks_plot(truevalue, predictedvalue)
  
  #---------------   Distribution of predicted probabilities values for true positives and true negatives   ---------------------------------  
  plot(0,0,type="n", xlim= c(0,1), ylim=c(0,7),     
       xlab="Prediction", ylab="Density",  
       main="Distribution of predicted probability for true positives and ture negatives")
  
  for (runi in 1:length(pred@predictions)) {
    lines(density(pred@predictions[[runi]][pred@labels[[runi]]==1]), col= "blue")
    lines(density(pred@predictions[[runi]][pred@labels[[runi]]==0]), col="green")
  } 
  
  #---------------   Cumulative Gains Chart   --------------------------------- 
  
  gain <- performance(pred,"tpr","rpp")
  gain1= performance(pred1,"tpr","rpp")
  
  plot(x=c(0, 1), y=c(0, 1), type="l", col="red", lwd=2,
       main="Cumulative Gains Chart",
       ylab="True Positive Rate", 
       xlab="Rate of Positive Predictions")
  
  
  gain.x = unlist(slot(gain, 'x.values'))
  gain.y = unlist(slot(gain, 'y.values'))
  
  lines(x=gain.x, y=gain.y, col="orange", lwd=2)
  
  gain1.x = unlist(slot(gain1, 'x.values'))
  gain1.y = unlist(slot(gain1, 'y.values'))
  
  lines(x=gain1.x, y=gain1.y, col="darkgreen", lwd=2)
  
  #---------------   Log Loss   --------------------------------- 
  
  Logloss1=LogLoss(fittedvalue, truevalue)
  
  
  return(list("Confusion Matrix and Statistics"=Confusionmatrix,
              "AUC"=auc1,
              "Optimum Threshhold"=Optcutoff,
              "Concordance Pairs"= Conpairs,
              "D Statistics"=Dstats,
              "K-S Statistics"=Ksstats,
              "Logloss"=Logloss1))
}


###################################################     Data Preparation    #######################################################

# Joining train and events dataframes based on display id using left join
setkey(clicks.train, display_id)
setkey(events, display_id)
result.1 <- merge(clicks.train, events, all.x=TRUE)
head(result.1)
str(result.1)
summary(result.1)


# Joining result.1 and promoted content based on ad id using left join
setkey(result.1,ad_id)
setkey(prom.cont,ad_id)
result.2 <- merge(result.1, prom.cont, all.x=TRUE)
head(result.2)
str(result.2)
summary(result.2)

# Joining result.2 and documents_meta based on document id using left join
result.2$document_id=result.2$document_id.x # using document id from events for further merge instead of promoted content
head(result.2)
summary(result.2)
setkey(result.2,document_id)
setkey(doc.meta,document_id)
result.3 <- merge(result.2,doc.meta, all.x=TRUE)
head(result.3)
summary(result.3)

# removing document.x and document.y columns
result.3[,c('document_id.x', 'document_id.y'):= NULL]
head(result.3)

# Joining result.3 and document topics based on document id using left join
setkey(result.3,document_id)
setkey(doc.enti,document_id)
result.4 <- merge(result.3, doc.enti, all.x=TRUE, allow.cartesian=TRUE)
head(result.4)
str(result.4)
summary(result.4)

############################################       Creating Test and Train Dataset      ##########################################
#  Creating test dataset
train.set=fread('D:/OU_SEMISTER_1/IDA/Project/sampled data/train.set.csv')
test.set=fread('D:/OU_SEMISTER_1/IDA/Project/sampled data/test.set.csv')
test.set= result.3[c(1:5000),]
head(test.set)
View(test.set)
summary(test.set)
write.csv(test.set,"test.set.csv")

# Creating train dataset
train.set= result.3[c(5001:25000),]
head(train.set)
View(train.set)
summary(train.set)
write.csv(train.set,"train.set.csv")

############################################      Feature Engineering             ####################################################

######################### Splitting the time ############

#??as.Date
#Newdate=as.Date(Result_pc_dm$publish_time,"%m/%d/%Y %H:%M")
#Newdate
#Hours <- format(as.POSIXct(strptime(Result_pc_dm$publish_time,"%m/%d/%Y %H:%M",tz="")) ,format = "%H:%M")

#Hours

#substring("Result_pc_dm$publish_time",seq())

#Dates <- format(as.POSIXct(strptime(weather$Time,"%d/%m/%Y %H:%M",tz="")) ,format = "%d/%d/%Y")
#output
#"27/27/2015" "23/23/2015" "31/31/2015" "20/20/2015" "23/23/2015" "31/31/2015"

#weather$Dates <- Dates
#weather$Hours <- Hours



# Feature engnieering on Train Dataset
train.set=fread('train.set.csv')
head(train.set)

#Removing unncessary comlumns
train.set= train.set[,-c(1,4,6,7,14)]

# Separating geo_location into separate columns
train.set$country <- sapply(train.set$geo_location, function(x) strsplit(x, '>')[[1]][1])
train.set$state <- sapply(train.set$geo_location, function(x) strsplit(x, '>')[[1]][2])
train.set$dma <- sapply(train.set$geo_location, function(x) strsplit(x, '>')[[1]][3])
head(train.set)
train.set= train.set[,-c(5)]
str(train.set)
summary(train.set)

#Missingness
aggr(train.set)
# if NA's found in summarry then add code for substitution

train.set$country[is.na(train.set$country)]="abc"
train.set$state[is.na(train.set$state)]="xyz"
train.set$dma[is.na(train.set$dma)]=0.995
  
# Applying one hot encoding
#train.set <- dummy.data.frame(train.set, names=c("document_id","ad_id","platform","campaign_id", "advertiser_id", "source_id", 
#                                                   "publisher_id", "country", "state", "dma"), sep="_")



# Feature engineering on Test dataset
test.set=fread('test.set.csv')

#Removing unncessary comlumns
test.set= test.set[,-c(1,4,6,7,14)]

# Separating geo_location into separate columns
test.set$country <- sapply(test.set$geo_location, function(x) strsplit(x, '>')[[1]][1])
test.set$state <- sapply(test.set$geo_location, function(x) strsplit(x, '>')[[1]][2])
test.set$dma <- sapply(test.set$geo_location, function(x) strsplit(x, '>')[[1]][3])
head(test.set)
test.set= test.set[,-c(5)]
summary(test.set)

#Missingness
aggr(test.set)
# if NA's found in summarry then add code for substitution

test.set$country[is.na(test.set$country)]="abc"
test.set$state[is.na(test.set$state)]="xyz"
test.set$dma[is.na(test.set$dma)]=0.995

head(test.set)
# Applying one hot encoding
#test.set <- dummy.data.frame(test.set, names=c("document_id","ad_id","platform","campaign_id", "advertiser_id", "source_id", "publisher_id",
#       "country", "state", "dma"), sep="_")


train.set$country[(train.set$country!="US"&train.set$country!="CA")]="Others"
str(train.set$dma)
train.set$dma <- as.numeric(as.character(train.set$dma))
str(train.set)
train.set=train.set[,-c("state","dma_1")]
head(train.set)

head(test.set)
test.set$country[(test.set$country!="US"&test.set$country!="CA")]="Others"
str(test.set$dma)
test.set$dma <- as.numeric(as.character(test.set$dma))
str(test.set)
test.set=test.set[,-c("state")]
head(test.set)
test.set1=test.set[,-c("country")]


train.set.1=train.set
train.set.2=train.set.1[,-c("country")]
head(train.set.2)
################################################Random Forest##########################################################


fit_rf <- randomForest(data=train.set.2, clicked ~ ., importance = T, ntrees=5000, mtry=3)
fit_rf

predfitrf=predict(fit_rf,newdata=train.set.2, type="class")
predfitrf
predfitrf=data.frame(predfitrf)
#predfitrf$Y

truevalue=train.set.2$clicked
fittedvalue=predfitrf
predictedvalue<-as.numeric(predfitrf>.5)


# Model evaluation using user-defined function
Model.eval(truevalue, fittedvalue, predictedvalue)

##############################################Document_categories###################################$####



cat3=doc.cate %>% 
  group_by(document_id)%>%
summarise(confidence_level= max(confidence_level)) 

cat4=as.data.frame(cat3)
  head(cat4)

write.csv(cat4,"doc.cate.max.conf.csv")

# set the ON clause as keys of the tables:
setkey(doc.cate,document_id,confidence_level)
setkey(doc.cate.max.conf,document_id,confidence_level)


# DO Fread

Result_2=doc.cate[, doc.cate[J(doc.cate.max.conf$document_id,doc.cate.max.conf$confidence_level)]]
View(doc.cate)
View(doc.cate.max.conf)
View(Result_2)

table(doc.cate$category_id)
table(doc.topics$topic_id)

doc.cate[ , .N , keyby=c("category_id")]



######################################SQL to R Interlinking##############################################
# Load library
library(RSQLite)

# Create a temporary directory
tmpdir <- tempdir()

# Set the file name
file <- "D:/OU_SEMISTER_1/IDA/Project/Data set/documents_categories.csv"


# Unzip the ONS Postcode Data file
unzip(file, exdir = tmpdir )

# Create a path pointing at the unzipped csv file
ONSPD_path <- paste0(tmpdir,"D:/OU_SEMISTER_1/IDA/Project/Data set/documents_categories.csv")

# Create a SQL Lite database connection
db_connection <- dbConnect(SQLite(), dbname="ons_lkp_db")

# Now load the data into our SQL lite database
dbWriteTable(conn = db_connection,
             name = "ONS_PD",
             value = ONSPD_path,
             row.names = FALSE,
             header = TRUE,
             overwrite = TRUE
)

# Check the data upload
dbListTables(db_connection)

---
Author: Ann Bowers
Date: January 11, 2019
Title: "Practical Machine Learning: Peer-graded Assignment"
output:
  html_document:
    keep_md: true
---



# Practical Machine Learning Course Project

#### Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions:  
 * __Class A__: Exactly according to the specification  
 * __Class B__: Throwing the elbows to the front  
 * __Class C__: Lifting the dumbbell only halfway   
 * __Class D__: Lowering the dumbbell only halfway  
 * __Class E__: Throwing the hips to the front  

Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience.  

More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The goal of your project is to predict the manner in each participant completed the did the exercise. 

####Loading the Data 
Load the training and testing data sets  


```r
library(caret)
library(elasticnet)
library(ada)
library(e1071)
library(klaR)

set.seed(3528) #for reproducibility

trainurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(url(trainurl), na.strings=c("NA","#DIV/0!",""))
testing<- read.csv(url(testurl), na.strings=c("NA","#DIV/0!",""))
```

####Data Preprocessing
Let's first look at the summary and structure of the training data set (Results are hidden due to their length).

```r
summary(training)
str(training)
```
From this list, we can see that the first 7 columns are not necessary for our analysis.  
We will also remove any fields that contain values with a variance near 0 or more than half of the values are NA.  


```r
#Remove first 7 variables
training<-training[,-(1:7)]

#Remove variables with variance near 0
nzVar <- nearZeroVar(training, saveMetrics = TRUE)
training <- training[, !nzVar$nzv]

#Remove variables with over half of the values equal to NA
halfNA <- sapply(colnames(training), function(x) if(sum(is.na(training[, x])) > 0.50*nrow(training))    {return(TRUE)
}else{
return(FALSE)
}
)
training <- training[, !halfNA]
```
After cleaning the data fields, we are left with the following variables


```r
names(training)
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```
   
Now that we have the data clean, we can separate it into a model training group and a model validation group (70%/30%, respectively).
    

```r
inTrain <- createDataPartition(training$classe, p = 0.80, list = FALSE)
train_set <- training[inTrain, ]
val_set <- training[-inTrain, ]
```
  
####Modeling
Because I am not familiar enough with the modeling types to pare down the models that I try, I will be testing results of several different models.  The linear model, additive logistic regression, and lasso methods will not work with this type of data.
1. Support Vector Machine - Linear (modsvm)
2. Random Forest (modrf)
3. Naive Bayes (modnb)
4. Linear Discriminate Analysis(modlda) 
5. LogitBoost (modlb)

   

```r
#preprocess using Principal Component Analysis to reduce model time and memory
tc <- trainControl(method = "cv", number = 7, verboseIter=FALSE , preProcOptions="pca", allowParallel=TRUE)
modsvm<-train(classe~., data = train_set, method="svmLinear", verbose = FALSE, trControl=tc)
modrf<-train(classe~., data = train_set, method="rf", verbose = FALSE, trControl=tc)
modnb<-train(classe~., data = train_set, method="nb", trControl=tc)
modlda<-train(classe~., data = train_set, method="lda", verbose = FALSE, trControl=tc)
modlb<-train(classe~., data = train_set, method="LogitBoost", verbose = FALSE, trControl=tc)
```
Now that the models have been created, we will compare the accuracy of each model to the others to determine the best model for our prediction.


```r
mod_type<-c("svm", "rf", "nb", "lda", "lb")
mod_acc<-c(median(modsvm$results$Accuracy), median(modrf$results$Accuracy), median(modnb$results$Accuracy), median(modlda$results$Accuracy), median(modlb$results$Accuracy) )
model_comp<-cbind(mod_type, mod_acc)
model_comp
```

```
##      mod_type mod_acc            
## [1,] "svm"    "0.77915651608532" 
## [2,] "rf"     "0.992865815368244"
## [3,] "nb"     "0.616855449352342"
## [4,] "lda"    "0.698834136031917"
## [5,] "lb"     "0.873088928754029"
```
It is clear that the Random Forest model creates the highest accuracy.  Therefore, we will use that to predict the validation set that we created earlier.  Then we will compare the predictions with the actual Activity Classes.


```r
rm(modsvm); rm(modnb); rm(modlda); rm(modlb)
predrf<- predict(modrf, val_set)
confusionMatrix(predrf, val_set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    2    0    0    0
##          B    1  757    5    0    0
##          C    0    0  679   10    1
##          D    0    0    0  633    6
##          E    0    0    0    0  714
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9936          
##                  95% CI : (0.9906, 0.9959)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9919          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9974   0.9927   0.9844   0.9903
## Specificity            0.9993   0.9981   0.9966   0.9982   1.0000
## Pos Pred Value         0.9982   0.9921   0.9841   0.9906   1.0000
## Neg Pred Value         0.9996   0.9994   0.9985   0.9970   0.9978
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1930   0.1731   0.1614   0.1820
## Detection Prevalence   0.2847   0.1945   0.1759   0.1629   0.1820
## Balanced Accuracy      0.9992   0.9977   0.9946   0.9913   0.9951
```
  
####Conclusion
  
From this information, we can see that Random Forest testing has created a model that is 99% accurate and has an out-of-sample error estimate of 0.78%.  Therefore, this is the model that we will use for this data.

####Testing

```r
pred_test<- predict(modrf, testing)
test_result<-data.frame(cbind(testing$problemid, pred_test))
View(test_result)
```


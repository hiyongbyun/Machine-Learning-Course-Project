# Machine Learning Course Project
### Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

### Exploratory Data Analyses
```{r echo=FALSE, message=FALSE, warning=FALSE, cache=TRUE}
library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
wleTrainingSource <- read_csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
```

Performing str() revealed a large number of variables with missing values. These variables should not be included in the prediction algorithm, along with the variables that indicate identifiers and timestamps (X1, user_name, raw_timestamp_part_1, raw_timestamp_part_2, num_window, cvtd_timestamp, new_window). Since "classe" is a character variable, it was transformed into a factor variable.

### Transforming the Dataset
```{r cache=TRUE}
varDelete <- NULL
for(i in 1:160) {
       if (mean(is.na(wleTrainingSource[,i])) > 0.9) {
               varDelete[i] <- names(wleTrainingSource[,i])
       }
}
varDelete <- na.omit(varDelete)
wleTrainingTidy <- select(wleTrainingSource, -varDelete)
wleTrainingTidy <- subset(wleTrainingTidy, select = -c(1:7))
wleTrainingTidy$classe <- as.factor(wleTrainingTidy$classe)
```

### Re-examining the Dataset
After removing variables with NAs occuring greater than 90% and the first 7 variables, the dataset was re-examined using the summary() function and checked for near zero variance predictors before proceeding with development of a model.

```{r cache=TRUE}
nearZeroVar(wleTrainingTidy, saveMetrics=TRUE)
```

### Partitioning into Training and Testing Datasets
Dataset was partitioned into 70% training and 30% testing.

```{r cache=TRUE}
set.seed(5336)
inTrain <- createDataPartition(wleTrainingTidy$classe, p=0.70, list=FALSE)
wleTraining <- wleTrainingTidy[inTrain,]
wleTesting <- wleTrainingTidy[-inTrain,]
```

### Developing a Model
Random forest was chosen to develop a prediction model for classifying 5 classes of exercise. All 52 variables were included in the model. randomForest() function was used for faster processing speed.
```{r cache=TRUE}
modelRF <- randomForest(classe~., data=wleTraining)
```

### Cross Validation and Accuracy
```{r cache=TRUE}
confusionMatrix(wleTesting$classe,predict(modelRF,wleTesting))
```
```
Confusion Matrix and Statistics
 
           Reference
Prediction    A    B    C    D    E
          A 1673    0    0    0    1
          B    3 1135    1    0    0
          C    0    6 1019    1    0
          D    0    0   15  949    0
          E    0    0    1    3 1078
 
Overall Statistics
                                           
                Accuracy : 0.9947          
                  95% CI : (0.9925, 0.9964)
     No Information Rate : 0.2848          
     P-Value [Acc > NIR] : < 2.2e-16       
                                           
                   Kappa : 0.9933          
                                           
Mcnemar's Test P-Value : NA              

Statistics by Class:
 
                      Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9982   0.9947   0.9836   0.9958   0.9991
Specificity            0.9998   0.9992   0.9986   0.9970   0.9992
Pos Pred Value         0.9994   0.9965   0.9932   0.9844   0.9963
Neg Pred Value         0.9993   0.9987   0.9965   0.9992   0.9998
Prevalence             0.2848   0.1939   0.1760   0.1619   0.1833
Detection Rate         0.2843   0.1929   0.1732   0.1613   0.1832
Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
Balanced Accuracy      0.9990   0.9969   0.9911   0.9964   0.9991
```

The model using the random forest algorithm was able to predict the testing dataset with 99.47% accuracy and near zero p-value. Out of sample error was calculated to be 0.53% (31 incorrect predictions out of 5885 observations).

### Reference
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

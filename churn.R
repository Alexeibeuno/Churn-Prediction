library(tidyverse)
#reading the dataset
churn <- read.csv("churn1.csv")

#to view the data
View(churn)

# To know the size of data
dim(churn)

#summary statistics for all the columns of the data
summary(churn)
str(churn)

#To check for null values
which(is.na(churn))
#There are no null values

#EDA
library(ggplot2)

#To check for null values
which(is.na(churn))
#There are no null values

ggplot(churn, aes(x = SeniorCitizen)) +
  geom_bar()
# There are less senior citizen  in the datasets

#chi-square test to know whether two categorical variables are dependent or not
chisq.test(churn$SeniorCitizen,churn$Churn)

#as both of them are dependent let's see how

#mosaic plot
counts <- table(churn$SeniorCitizen, churn$Churn)
mosaicplot(counts, xlab='Senior Citizen', ylab ='Churn',
           main='Churn by SeniorCitizen', col='orange')

hist(churn$tenure ,labels = TRUE,col=blues9,xlim = c(0,70)) 
#After looking at the above histogram we can see that a lot of 
#customers have been with the telecom company for just a month to 6 months, 
#while quite a many are there for about 72 months.

ggplot(churn, aes(x = group)) +
  geom_bar(color="blue", fill=rgb(0.1,0.3,0.5,0.8))
# mostly people have 12 months contract

ggplot(churn, aes(x = Contract,fill=as.factor(Contract) )) +
  geom_bar() + scale_fill_hue(c = 30) +
  theme(legend.position="none")
#As we can see from this graph most of the customers are in the month to month contract. While there are equal number of customers in the 1 year and 2 year contracts.


ggplot(churn, aes(x = Churn)) +
  geom_bar(color="blue", fill=rgb(0.1,0.3,0.5,0.8))
# There are less number of people who are switching to other service


plot(churn$MonthlyCharges, churn$TotalCharges,
     xlim=c(0,100) , ylim=c(0,8000), 
     pch=15, 
     cex=0.5, 
     col="#69b3a2",
     xlab="Monthly Charges", ylab="Total Charges",
     main="Relation b/w Monthly and Total Charges"
)
#the total charges increases as the monthly bill for a customer increases

ggplot(churn, aes(x = PaymentMethod,fill=as.factor(PaymentMethod) )) +
  geom_bar() + scale_fill_hue(c = 30) +
  theme(legend.position="none")
# The most used payment method is electronic check

#Model Building

#To check the dependence of categorical variables with the target variable

#Chi-Square test of independence
#H0: The two variables are independent
#H1: The two variables are not independent
#We reject the null hypothesis when the p-value is greater than 0.05  

chisq.test(churn$gender,churn$Churn)
chisq.test(churn$SeniorCitizen,churn$Churn)
chisq.test(churn$Partner,churn$Churn)
chisq.test(churn$Dependents,churn$Churn)
chisq.test(churn$PhoneService,churn$Churn)
chisq.test(churn$MultipleLines,churn$Churn)
chisq.test(churn$InternetService,churn$Churn)
chisq.test(churn$OnlineSecurity,churn$Churn)
chisq.test(churn$OnlineBackup,churn$Churn)
chisq.test(churn$DeviceProtection,churn$Churn)
chisq.test(churn$TechSupport,churn$Churn)
chisq.test(churn$StreamingTV,churn$Churn)
chisq.test(churn$StreamingMovies,churn$Churn)
chisq.test(churn$Contract,churn$Churn)
chisq.test(churn$PaperlessBilling,churn$Churn)
chisq.test(churn$PaymentMethod,churn$Churn)

#Since the p-value for Gender, PhoneService is greater than 0.05, we reject the 
#null hypothesis, hence these variables are not dependent on Churn(the target variable)
#Hence, we keep these two variables, and discard the rest.

#Correlation matrix to check dependence of continuous variables
x <- data.frame(churn$tenure, churn$MonthlyCharges, churn$TotalCharges)
cor(x)

#Total charges is highly correlated with other variables
#The only variables that are not correlated with the others are, Gender, PhoneService, Tenure and Monthly Charges

#Label Encoding in R
#install.packages("superml")
library(superml)
label <- LabelEncoder$new()
churn$Churn <- label$fit_transform(churn$Churn)
churn$gender 
View(churn)

#Building the Logistic Regression model
lm = glm(Churn~ gender+PhoneService+tenure+MonthlyCharges , data=churn, 
         family=binomial(logit))
summary(lm) 


#calculate VIF values for each predictor in our model
car::vif(lm)
#Since the VIF values are below 5, it indicates that there is no multicollinearity

#To determine the optimal cutoff
predicted <- predict(lm)
library(InformationValue)
cutoff <- optimalCutoff(churn$Churn, predicted)[1]
cutoff

#Any individual with a probability of 0.168 or higher is predicted to churn

confusionMatrix(churn$Churn, predicted)

#Sensitivity or true positive rate
sensitivity(churn$Churn, predicted)

#Specificity or true negative rate
specificity(churn$Churn, predicted)

#total mis-classification error rate
misClassError(churn$Churn, predicted, threshold=cutoff)


library(pROC)
#ROC
plotROC(churn$Churn, predicted)

#Since AUC is quite high, this indicates that the model is good


cutoffs = seq(0,1,0.001)
cutoffs

library(MLmetrics)
auc_values = c()
sensitivities_values = c()
specificity_values = c()
f1scores = c()
for(i in cutoffs){
  predicted <- predict(lm,type='response')
  predicted = ifelse(predicted<i,0,1)
  roc_object <- roc( churn$Churn , predicted) 
  auc_values = append(auc_values,auc(roc_object))
  sensitivities_values <- append(sensitivities_values,sensitivity(churn$Churn, predicted))
  specificity_values <- append(specificity_values,specificity(churn$Churn, predicted))
  f1scores <- append(f1scores,F1_Score(predicted,churn$Churn))
}

check <- data.frame(auc=auc_values,sensitity = sensitivities_values,
                    f1Scores = f1scores,
                    specificity=specificity_values, cutoff = cutoffs)
check

write.csv(check,"check.csv")
plot(x=check$cutoff,y=check$auc)
plot(x=check$cutoff,y=check$f1scores)



#--------------------------------------------------------------------------------
                         #SUPPORT VECTOR MACHINE

#load the package e1071 which contains the svm function 
library("e1071")

library('caTools')

churn$Churn <- as.factor(churn$Churn)

sel <- sample.split(churn$Churn,SplitRatio = 0.7)

sel

churn_train <- subset(churn,sel==TRUE)
churn_test <- subset(churn,sel==FALSE)

svmfit_l <- svm(Churn~ gender+PhoneService+tenure+MonthlyCharges,data=churn_train,kernel="linear")

svmfit_r <- svm(Churn~ gender+PhoneService+tenure+MonthlyCharges,data=churn_train,kernel="radial")


# FOR LINEAR SVM

#Printing the svmfit gives its summary.
print(svmfit_l)

churn_test$churn_pred <- predict(svmfit_l,churn_test)
View(churn_test)

# find the accuracy

library('caret')

confusionMatrix(table(churn_test$Churn,churn_test$churn_pred))

# FOR RADIAL SVM

#Printing the svmfit gives its summary.
print(svmfit_r)

churn_test$churn_pred <- predict(svmfit_r,churn_test)
View(churn_test)

# find the accuracy

confusionMatrix(table(churn_test$Churn,churn_test$churn_pred))


#--------------------------------------------------------------------------------
                            #DECISION TREE

churn$Churn <- as.factor(churn$Churn)
sel <- sample.split(churn$Churn,SplitRatio = 0.7)
sel
churn_train <- subset(churn,sel==TRUE)
churn_test <- subset(churn,sel==FALSE)

library(rpart)
#install.packages('rpart.plot')
library(rpart.plot)

#decision tree diagram
arbre=rpart(Churn~gender+PhoneService+tenure+MonthlyCharges,data=churn_train,method="class")
rpart.plot(arbre)

#confusion metrix
prevarbre=predict(arbre,newdata=churn_test,type="prob")
previsions2=ifelse(prevarbre[,2]>0.32,"Yes","No")
table(previsions2,churn_test$Churn)

#accuracy
CM_arbre<-table(prevarbre[,2]>0.32,churn_test$Churn)
accuracy_arbre=(sum(diag(CM_arbre)))/sum(CM_arbre)
accuracy_arbre

#roc plot
plot(roc(predictor=prevarbre[,2],response=churn_test$Churn)) #ROC curve

#-------------------------------------------------------------------------------
                              #PCA(Principal Component Analysis)
#Pca can be applied only on numerical variables
#calculate principal components
num_churn <- data.frame(churn$tenure, churn$MonthlyCharges, churn$TotalCharges)
comp <- prcomp(num_churn, scale=TRUE)

#reverse the signs
comp$rotation <- -1*comp$rotation

#display principal components
comp$rotation

#We can see that the first principal component has high variation, which indicates that it is significant
#Second Principal component shows high value for Monthly charges
#principal component scores are stored in comp$x, we multiply these scores by 1 to reverse signs

comp$x <- -1*comp$x

#Display the first six scores
head(comp$x)

#display 
head(churn[order(-churn$tenure),])

#Find the variation explained by each principal component
comp$sdev^2 / sum(comp$sdev^2)

#First principal component explains 72.66% of the variation
#Second principal component explains 25.36% of the variation
#Third principal component explains 1.98% of the variation
#Only the first two explain the majority of the variance in the data

#Create a screeplot - a plot that displays the total variance explained by each principal component
var_explained <- comp$sdev^2 / sum(comp$sdev^2)

#scree plot
qplot(c(1:3), var_explained) + 
  geom_line() +
  xlab("Principal Component") +
  ylab("Scree Plot") +
  ylim(0, 1)

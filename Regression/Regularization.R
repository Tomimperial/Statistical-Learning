library(caret)
library(glmnet)
library(mlbench)
library(psych)
library(corrplot)
require(Amelia)
require(stargazer)
library(xtable)
require(sqldf)



#Loading the boston data

data('BostonHousing')
data<-BostonHousing
str(data)
data$chas<-as.numeric(data$chas)

# Priliminary Data Analaysis
M<-cor(data)
corrplot(M, method="circle")
missmap(data,col=c('blue','black'),y.at=1,y.labels='',legend=TRUE)

#Data Splitting

set.seed(123)

train_ind <- sample(1:nrow(data), size = floor(0.70 * nrow(data)))
training<-data[train_ind,]
testing<-data[-train_ind,]
#PART -1 

histogram(data$crim)
histogram(log(data$crim))



# 1.1 Linear Regression with no transformation

custom<-trainControl(method='repeatedcv',number=10,repeats=5)
lm<-train(crim~.,training,method='lm',trControl=custom)
lm$results
xtable(summary(lm), type = "latex", file = "filename2.tex")
summary(lm)
plot(lm$finalModel,1)
plot(lm$finalModel,2)

lm$finalModel
xtable(lm$finalModel, type = "latex", file = "filename2.tex")
testing$predict<-predict(lm$finalModel,testing)

P1<-data.frame(
  R2 = R2(testing$predict, testing$crim),
  RMSE = RMSE(testing$predict, testing$crim),
  MAE = MAE(testing$predict, testing$crim)
)

# 1.2 Ridge Regression 

ridge<-train(crim~.,training,method='glmnet',tuneGrid=expand.grid(alpha=0,lambda=seq(0.01,1.6,length=10)),trControl=custom)


jpeg("ridge_lambda.jpeg", width = 4, height = 4, units = 'in', res = 300)
plot(ridge)
dev.off()

ridge$bestTune
ridge$finalmodel
ridge$results
xtable(ridge$results, type = "latex", file = "filename2.tex")

jpeg("ridge_dev.jpeg", width = 4, height = 4, units = 'in', res = 300)
plot(ridge$finalModel,xvar='dev',label=TRUE)
dev.off()

plot(ridge$finalModel,xvar='lambda',label=TRUE)
plot(ridge$finalModel,xvar='dev',label=TRUE)
plot(varImp(ridge,scale=F))

ridge$bestTune
x1<-coef(ridge$finalModel,s=ridge$bestTune$lambda)
x1

testing$predict<-predict(object=ridge,testing[ , ! colnames(training) %in% c('lncrim') ])

P2<-data.frame(
  R2 = R2(testing$predict, testing$crim),
  RMSE = RMSE(testing$predict, testing$crim),
  MAE = MAE(testing$predict, testing$crim)
)

# 1.3 Lasso Regression 

lasso<-train(crim~.,training,method='glmnet',tuneGrid=expand.grid(alpha=1,lambda=seq(0.0001,1,length=10)),trControl=custom)

jpeg("lasso_lambda.jpeg", width = 4, height = 4, units = 'in', res = 300)
plot(lasso)
dev.off()

jpeg("lasso_dev.jpeg", width = 4, height = 4, units = 'in', res = 300)
plot(lasso$finalModel,xvar='dev',label=TRUE)
dev.off()

plot(lasso)

lasso

plot(lasso$finalModel,xvar='lambda',label=TRUE)
plot(lasso$finalModel,xvar='dev',label=TRUE)
plot(varImp(lasso,scale=F))

x1<-coef(lasso$finalModel,s=lasso$bestTune$lambda)
x1

testing$predict<-predict(object=lasso,testing[ , ! colnames(training) %in% c('lncrim') ])

P3<-data.frame(
  R2 = R2(testing$predict, testing$crim),
  RMSE = RMSE(testing$predict, testing$crim),
  MAE = MAE(testing$predict, testing$crim)
)

# 1.4 Elastic Net Regression 

en<-train(crim~.,training,method='glmnet',tuneGrid=expand.grid(alpha=seq(0,1,length=20),lambda=seq(0.0001,1.1,length=25)),trControl=custom)
plot(en)
en

jpeg("en_lambda2.jpeg", width = 4, height = 4, units = 'in', res = 300)
plot(en$finalModel,xvar='lambda',label=TRUE)
dev.off()

jpeg("en_dev.jpeg", width = 4, height = 4, units = 'in', res = 300)
plot(en$finalModel,xvar='dev',label=TRUE)
dev.off()

plot(en$finalModel,xvar='lambda',label=TRUE)
plot(en$finalModel,xvar='dev',label=TRUE)
plot(varImp(en,scale=F))
en$bestTune
best<-en$finalModel
x1<-coef(best,s=en$bestTune$lambda)
x1
testing$predict<-predict(object=en,testing[ , ! colnames(training) %in% c('lncrim') ])

P4<-data.frame(
  R2 = R2(testing$predict, testing$crim),
  RMSE = RMSE(testing$predict, testing$crim),
  MAE = MAE(testing$predict, testing$crim)
)


#1.5 Summary

model_list<-list(LinearModel=lm,Ridge=ridge,Lasso=lasso,ElasticNet=en)
res<-resamples(model_list)
xx<-summary(res)
xx$statistics$Rsquared
xtable(xx$statistics$Rsquared, type = "latex", file = "filename2.tex")
stargazer(summary(res))

jpeg("res2.jpeg", width = 8, height = 4, units = 'in', res = 300)
bwplot(res)
dev.off()


test_accu_summ<-rbind(P1,P2,P3,P4)

#PART 2
#adding new transformed variable
data('BostonHousing')
data<-BostonHousing
str(data)
data$chas<-as.numeric(data$chas)
data$lncrim<-log(data$crim)
data<-data[ , ! colnames(data) %in% c('crim') ]

set.seed(123)
train_ind <- sample(1:nrow(data), size = floor(0.70 * nrow(data)))
training<-data[train_ind,]
testing<-data[-train_ind,]

# 2.1 Linear Regression with variable transformation

custom<-trainControl(method='repeatedcv',number=10,repeats=5)
lm<-train(lncrim~.,training,method='lm',trControl=custom)
lm$results
summary(lm)
xtable(lm$results, type = "latex", file = "filename2.tex")
plot(lm$finalModel,1)

jpeg("Plots_err_LR_2_a.jpeg", width = 8, height = 4, units = 'in', res = 300)
plot(lm$finalModel,2)
dev.off()

plot(lm$finalModel,2)
lm$finalModel
testing$predict<-predict(lm$finalModel,testing)

P1<-data.frame(
  R2 = R2(testing$predict, testing$lncrim),
  RMSE = RMSE(testing$predict, testing$lncrim),
  MAE = MAE(testing$predict, testing$lncrim)
)

# 2.2 Ridge Regression


ridge<-train(lncrim~.,training,method='glmnet',tuneGrid=expand.grid(alpha=0,lambda=seq(0.0001,0.2,length=30)),trControl=custom)

jpeg("ridge2_dev.jpeg", width = 4, height = 4, units = 'in', res = 300)
plot(ridge$finalModel,xvar='dev',label=TRUE)
dev.off()

plot(ridge)

require(ridge)
pvals(ridge$finalModel)
xtable(ridge$results, type = "latex", file = "filename2.tex")

plot(ridge$finalModel,xvar='lambda',label=TRUE)
plot(ridge$finalModel,xvar='dev',label=TRUE)
plot(varImp(ridge,scale=F))

x1<-coef(ridge$finalModel,s=ridge$bestTune$lambda)
x1

testing$predict<-predict(object=ridge,testing[ , ! colnames(training) %in% c('lncrim') ])

P2<-data.frame(
  R2 = R2(testing$predict, testing$lncrim),
  RMSE = RMSE(testing$predict, testing$lncrim),
  MAE = MAE(testing$predict, testing$lncrim)
)

# 2.3 Lasso Regression

lasso<-train(lncrim~.,training,method='glmnet',tuneGrid=expand.grid(alpha=1,lambda=seq(0.0001,0.02,length=10)),trControl=custom)

jpeg("lasso2_dev.jpeg", width = 4, height = 4, units = 'in', res = 300)
plot(lasso$finalModel,xvar='dev',label=TRUE)
dev.off()

plot(lasso)

lasso$results
xtable(lasso$results, type = "latex", file = "filename2.tex")

plot(lasso$finalModel,xvar='lambda',label=TRUE)
plot(lasso$finalModel,xvar='dev',label=TRUE)
plot(varImp(lasso,scale=F))

testing$predict<-predict(object=lasso,testing[ , ! colnames(training) %in% c('lncrim') ])

x1<-coef(lasso$finalModel,s=lasso$bestTune$lambda)
x1

P3<-data.frame(
  R2 = R2(testing$predict, testing$lncrim),
  RMSE = RMSE(testing$predict, testing$lncrim),
  MAE = MAE(testing$predict, testing$lncrim)
)

# 2.4 Elastic Net Regression

en<-train(lncrim~.,training,method='glmnet',tuneGrid=expand.grid(alpha=seq(0.6,1,length=20),lambda=seq(0.0001,0.02,length=25)),trControl=custom)
plot(en)
en
jpeg("en2_lambda.jpeg", width = 8, height = 4, units = 'in', res = 300)
plot(en)
dev.off()

jpeg("en2_dev.jpeg", width = 4, height = 4, units = 'in', res = 300)
plot(en$finalModel,xvar='dev',label=TRUE)
dev.off()

plot(en$finalModel,xvar='lambda',label=TRUE)
plot(en$finalModel,xvar='dev',label=TRUE)
plot(varImp(en,scale=F))
en$bestTune
best<-en$finalModel
coef(best,s=en$bestTune$lambda)

testing$predict<-predict(object=en,testing[ , ! colnames(training) %in% c('lncrim') ])

P4<-data.frame(
  R2 = R2(testing$predict, testing$lncrim),
  RMSE = RMSE(testing$predict, testing$lncrim),
  MAE = MAE(testing$predict, testing$lncrim)
)


#2.5 Summary

model_list<-list(LinearModel=lm,Ridge=ridge,Lasso=lasso,ElasticNet=en)
res<-resamples(model_list)
xx<-summary(res)
xx$statistics
xtable(xx$statistics$Rsquared, type = "latex", file = "filename2.tex")
stargazer(summary(res))

jpeg("res2.jpeg", width = 8, height = 4, units = 'in', res = 300)
bwplot(res)
dev.off()



test_accu_summ<-rbind(P1,P2,P3,P4)
xtable(test_accu_summ, type = "latex", file = "filename2.tex")





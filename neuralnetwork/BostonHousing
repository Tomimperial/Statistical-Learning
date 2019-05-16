library(mlbench)
library(h2o)
library(lattice)
data('BostonHousing')
dataf<-BostonHousing
dataf$chas<-as.numeric(dataf$chas)

h2o.init()
# Remove running clusters to be more efficient
h2o.removeAll()
# Convert data frame to h2o enviornment
df<-as.h2o(dataf)
# The following code split the data into test and train on 60% train 20% for validation
# and 20% test
splits <- h2o.splitFrame(
  df,           
  c(0.6,0.2),   ##  create splits of 60% and 20%; 
  seed=189)
# The split is used to get train data
train <- h2o.assign(splits[[1]], "train.hex")   ## R train, H2O train.hex
valid <- h2o.assign(splits[[2]], "valid.hex")   ## R valid, H2O valid.hex
test <- h2o.assign(splits[[3]], "test.hex")     ## R test, H2O test.hex

search_criteria <- list(strategy = "RandomDiscrete", 
                        max_models = 100,
                        max_runtime_secs = 900,
                        stopping_tolerance = 0.001,
                        stopping_rounds = 15,
                        seed = 42)

hyper_params <- list(
  hidden = c(140,130,120,110,100,90,80,70,60,50,40,30,25,20,15,10,5)
)
nn_grid <- h2o.grid(
  algorithm = "deeplearning", # Neural Network Model
  grid_id = "dl_grid",     # Model Name
  training_frame=train,       # training data
  validation_frame=valid,     # validation data 
  x=2:14,                     # Predictors
  y=1,                       # dependent variable
  nfolds = 5,# No of hidden layers and hidden nodes
  seed=300,
  activation='Tanh',
  hyper_params = hyper_params,
  standardize = TRUE# number of runs
)




x<-h2o.getGrid("dl_grid",sort_by='r2')@summary_table
x_dash<-sapply(x$hidden,function(x) gsub("\\[|\\]", "", x))
x$hidden<-as.numeric(x_dash)
x$r2<-as.numeric(x$r2)
x$r2<-round()
plot(x$hidden,x$r2)

library('ggplot2')
jpeg("MLP1.jpeg", width = 8, height = 4, units = 'in', res = 300)
ggplot(x, aes(x=hidden, y=r2)) +
  geom_line(col='red') + 
  geom_point() + ylim(0.4, 0.55)+geom_text(aes(label=hidden),hjust=0, vjust=0)
dev.off()



splits <- h2o.splitFrame(
  df,           
  c(0.5,0.3),   ##  create splits of 60% and 20%; 
  seed=103)

nn_model <- h2o.deeplearning( # Neural Network Model
  model_id="dl_model_nn",     # Model Name
  training_frame=train,       # training data
  validation_frame=valid,     # validation data 
  x=2:14,                     # Predictors
  y=1,                       # dependent variable
  hidden=c(30), 
  activation='Tanh',
  nfolds=5,# No of hidden layers and hidden nodes
  epochs=100,
  export_weights_and_biases=T# number of runs
)

jpeg("MD1.jpeg", width = 8, height = 4, units = 'in', res = 300)
h2o.varimp_plot(nn_model)
dev.off()

plot(nn_model)

jpeg("MLP2.jpeg", width = 8, height = 4, units = 'in', res = 300)
plot(nn_model)
dev.off()

# Train data performance
nn_train_perf<-h2o.performance(h2o.getModel('dl_model_nn'))
nn_test_perf<-h2o.performance(h2o.getModel('dl_model_nn'), newdata = test)
nn_train_perf@metrics$RMSE
nn_test_perf@metrics$RMSE
nn_train_perf@metrics$r2
nn_test_perf@metrics$r2


nn_sum<-summary(nn_model) 


weights1 <-as.data.frame( h2o.weights(nn_model,matrix_id=1))
weights2<-as.data.frame( h2o.weights(nn_model,matrix_id=2))
biases1 <- as.data.frame(h2o.biases(nn_model,vector_id=1))
biases2 <- as.data.frame(h2o.biases(nn_model,vector_id=2))

W1<-cbind(biases1,weights1)
W2<-cbind(biases2,weights2)
w<-c(as.vector(t(W1)),as.vector(t(W2)))
     require(NeuralNetTools)
jpeg("MLP3.jpeg", width = 8, height = 4, units = 'in', res = 300)
plotnet(w, struct = c(13,30,1),x_names = names(dataf[-1]),y_names='crim')
dev.off()

L<-data.frame()
# Lasso
k=1
for (i in c(0.000001,0.000015,0.00001,0.00015,0.0001,0.001,0.01)){
  nn_model <- h2o.deeplearning( # Neural Network Model
    model_id="dl_model_nn",     # Model Name
    training_frame=train,       # training data
    validation_frame=valid,     # validation data 
    x=2:14,                     # Predictors
    y=1,                       # dependent variable
    hidden=c(30), 
    activation='Tanh',
    nfolds=5,
    loss = 'Quadratic',# No of hidden layers and hidden nodes
    epochs=2000,
    l1=i,
    seed=233
  ) 
  nn_train_perf<-h2o.performance(h2o.getModel('dl_model_nn'))
  nn_test_perf<-h2o.performance(h2o.getModel('dl_model_nn'), newdata = valid)
  print(i)
  print(nn_train_perf@metrics$RMSE)
  print(nn_train_perf@metrics$r2)
  print(nn_test_perf@metrics$RMSE)
  print(nn_test_perf@metrics$r2)
  L[1,k]=i
  L[2,k]=nn_train_perf@metrics$RMSE
  L[3,k]=nn_train_perf@metrics$r2
  L[4,k]=nn_test_perf@metrics$RMSE
  L[5,k]=nn_test_perf@metrics$r2
  k=k+1
}
plot_L1<-data.frame(t(L))
require(ggplot2)

jpeg("MLP44.jpeg", width = 8, height = 4, units = 'in', res = 300)
ggplot(data=plot_L1, aes(x=X1, y=X4,col='blue')) + 
  geom_line(size=2) + 
  scale_x_log10() + 
  scale_y_log10() + 
  xlab('L1 Regularization parameter') + 
  ylab("RMSE-validation")
dev.off()

nn_model1 <- h2o.deeplearning( # Neural Network Model
  model_id="dl_model_L",     # Model Name
  training_frame=train,       # training data
  validation_frame=valid,     # validation data 
  x=2:14,                     # Predictors
  y=1,                       # dependent variable
  hidden=c(30), 
  activation='Tanh',
  nfolds=5,# No of hidden layers and hidden nodes
  epochs=100,
  l1=0.00015,
  export_weights_and_biases=T# number of runs
)

jpeg("MD2.jpeg", width = 8, height = 4, units = 'in', res = 300)
h2o.varimp_plot(nn_model1)
dev.off()

nn_train_perf<-h2o.performance(h2o.getModel('dl_model_L'))
nn_train_perf
nn_test_perf<-h2o.performance(h2o.getModel('dl_model_L'), newdata = test)
nn_test_perf
nn_train_perf@metrics$RMSE
nn_test_perf@metrics$RMSE
nn_train_perf@metrics$r2
nn_test_perf@metrics$r2

# Ridge
L<-data.frame()
k=1
for (i in c(0.0001,0.00015,0.001,0.0015,0.01)){
  nn_model <- h2o.deeplearning( # Neural Network Model
    model_id="dl_model_nn",     # Model Name
    training_frame=train,       # training data
    validation_frame=valid,     # validation data 
    x=2:14,                     # Predictors
    y=1,                       # dependent variable
    hidden=c(30), 
    activation='Tanh',
    nfolds=5,
    loss = 'Quadratic',# No of hidden layers and hidden nodes
    epochs=2000,
    l2=i,
    seed=233
  ) 
  nn_train_perf<-h2o.performance(h2o.getModel('dl_model_nn'))
  nn_test_perf<-h2o.performance(h2o.getModel('dl_model_nn'), newdata = valid)
  print(i)
  print(nn_train_perf@metrics$RMSE)
  print(nn_train_perf@metrics$r2)
  print(nn_test_perf@metrics$RMSE)
  print(nn_test_perf@metrics$r2)
  L[1,k]=i
  L[2,k]=nn_train_perf@metrics$RMSE
  L[3,k]=nn_train_perf@metrics$r2
  L[4,k]=nn_test_perf@metrics$RMSE
  L[5,k]=nn_test_perf@metrics$r2

  k=k+1
}

plot_L1<-data.frame(t(L))
require(ggplot2)
jpeg("MLP55.jpeg", width = 8, height = 4, units = 'in', res = 300)
ggplot(data=plot_L1, aes(x=X1, y=X4)) + geom_point()+
  geom_line(size=2) + 
  scale_x_log10() + 
  scale_y_log10() + 
  xlab('L2 Regularization parameter') + 
  ylab("RMSE-validation")
dev.off()

nn_model2 <- h2o.deeplearning( # Neural Network Model
  model_id="dl_model_R",     # Model Name
  training_frame=train,       # training data
  validation_frame=valid,     # validation data 
  x=2:14,                     # Predictors
  y=1,                       # dependent variable
  hidden=c(30), 
  activation='Tanh',
  nfolds=5,# No of hidden layers and hidden nodes
  epochs=100,
  l2=0.001,
  export_weights_and_biases=T# number of runs
)

jpeg("MD3.jpeg", width = 8, height = 4, units = 'in', res = 300)
h2o.varimp_plot(nn_model2)
dev.off()

nn_train_perf<-h2o.performance(h2o.getModel('dl_model_R'))
nn_train_perf
nn_test_perf<-h2o.performance(h2o.getModel('dl_model_R'), newdata = test)
nn_test_perf
nn_train_perf@metrics$RMSE
nn_test_perf@metrics$RMSE
nn_train_perf@metrics$r2
nn_test_perf@metrics$r2
# Elastic Net

L<-data.frame()
k=1
for (j in c(0.000001,0.000015,0.00001,0.00015,0.00015,0.0001,0.0015,0.001,0.015,0.01)){
  for (i in c(0.000001,0.000015,0.00001,0.00015,0.00015,0.0001,0.0015,0.001,0.015,0.01)){
    nn_model <- h2o.deeplearning( # Neural Network Model
      model_id="dl_model_nn",     # Model Name
      training_frame=train,       # training data
      validation_frame=valid,     # validation data 
      x=2:14,                     # Predictors
      y=1,                       # dependent variable
      hidden=c(30), 
      activation='Tanh',
      nfolds=5,
      loss = 'Quadratic',# No of hidden layers and hidden nodes
      epochs=2000,
      l2=i,
      seed=233
    ) 
    nn_train_perf<-h2o.performance(h2o.getModel('dl_model_nn'))
    nn_test_perf<-h2o.performance(h2o.getModel('dl_model_nn'), newdata = valid)
    print(i)
    print(nn_train_perf@metrics$RMSE)
    print(nn_train_perf@metrics$r2)
    print(nn_test_perf@metrics$RMSE)
    print(nn_test_perf@metrics$r2)
    L[k,1]=j
    L[k,2]=i
    L[k,3]=nn_train_perf@metrics$RMSE
    
    k=k+1
  }}



nn_model <- h2o.deeplearning( # Neural Network Model
  model_id="dl_model_nn",     # Model Name
  training_frame=train,       # training data
  validation_frame=valid,     # validation data 
  x=2:14,                     # Predictors
  y=1,                       # dependent variable
  hidden=c(30), 
  activation='Tanh',
  l2=0.01,
  nfolds=5,# No of hidden layers and hidden nodes
  epochs=1000
)
L2<-L
L2$V1<-log(L2$V1)
L2$V2<-log(L2$V2)
L2$V3<-log(L2$V3)
jpeg("MLP_b.jpeg", width = 8, height = 4, units = 'in', res = 300)
wireframe(V3 ~ V1*V2, data = L2,
          xlab = "L1 regularization", ylab = "L2 Regularization",zlab='RMSE',
          drape = TRUE,
          colorkey = TRUE,
          screen = list(z = 45, x = -65)
)

nn_model3 <- h2o.deeplearning( # Neural Network Model
  model_id="dl_model_E",     # Model Name
  training_frame=train,       # training data
  validation_frame=valid,     # validation data 
  x=2:14,                     # Predictors
  y=1,                       # dependent variable
  hidden=c(30), 
  activation='Tanh',
  nfolds=5,# No of hidden layers and hidden nodes
  epochs=100,
  l2=0.001,
  l1=0.0001,
  export_weights_and_biases=T# number of runs
)

jpeg("MD4.jpeg", width = 8, height = 4, units = 'in', res = 300)
h2o.varimp_plot(nn_model3)
dev.off()

nn_train_perf<-h2o.performance(h2o.getModel('dl_model_E'))
nn_train_perf
nn_test_perf<-h2o.performance(h2o.getModel('dl_model_E'), newdata = test)
nn_test_perf
nn_train_perf@metrics$RMSE
nn_test_perf@metrics$RMSE
nn_train_perf@metrics$r2
nn_test_perf@metrics$r2

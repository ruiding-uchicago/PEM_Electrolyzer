options(java.parameters = "-Xmx24g")
library(xlsx)
library(readxl)
library(hydroGOF)
library(randomForest)
library(ggplot2)
library(circlize)
library(RColorBrewer)
library(dplyr)
library(randomForestExplainer)
library(pdp)
library(tcltk)
library(patchwork)
library(caret)
library(ggrepel)
library(data.table)
library(ggraph)
library(igraph)
library(tidyverse)
library(RColorBrewer) 
library(pdp)
library(Rcpp)
library(randomForest)
library(randomForestExplainer)
library(caret)
library(networkD3)
library(shiny)
library(tidyverse)
library(xgboost)
library(lightgbm)
library(EIX)
library(Matrix)

colindex<-c('OT',	'FR',	'AA',	'IR',	'RU',	'O',	'ICA',	'PT',	'ICC',	'AL',	'CL',	'MT',	'EW',	'CD',	'ST','CL_10')
r2_general <-function(preds,actual){ 
  return(1- sum((preds - actual) ^ 2)/sum((actual - mean(actual))^2))
}
nthmax<-function(x,n){
  y<-as.numeric(x)
  y<-order(y,decreasing=TRUE)
  return(x[y[n]])
}
##################################################
data_PC=read.csv("ST10_data.csv",head=T)
data_PC_train=read.csv("ST10_train.csv",head=T)
data_PC_test=read.csv("ST10_test.csv",head=T)
#set.seed(9)
#train <- sample(nrow(data_PC), 0.85*nrow(data_PC))
#data_PC_train=data_PC[train,]
#data_PC_test=data_PC[-train,]
#head(data_PC_train)
#head(data_PC_test)
num_trees<-200
##################################################
set.seed(3)

model_matrix_all<-model.matrix(CL_10 ~ . - 1, data_PC)
model_martix_train <- model.matrix(CL_10 ~ . - 1, data_PC_train)
model_martix_test <- model.matrix(CL_10 ~ . - 1, data_PC_test)
data_train <- xgb.DMatrix(model_martix_train, label = data_PC_train$CL_10)
data_test <- xgb.DMatrix(model_martix_test, label = data_PC_test$CL_10)

param <- list(max_depth = 9, eta = 0.1, subsample=0.85,silent = 1,lambda=0.001,alpha=0.01,objective = "binary:logistic")

xgb_model <- xgb.train(param, data_train, nrounds = num_trees)

xgb_model

lolli<-lollipop(xgb_model,model_matrix_all)
plot(lolli,labels='topAll',log_scale=T,threshold = 0.2)

imp<-importance(xgb_model,model_matrix_all,option='both')
plot(imp,top = 15)
inter<-interactions(xgb_model,model_matrix_all,option='interactions')
plot(inter)
inter2<-interactions(xgb_model,model_matrix_all,option='pairs')
plot(inter2)


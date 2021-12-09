options(java.parameters = "-Xmx24g")

library(xgboost)
library(lightgbm)
library(EIX)
library(Matrix)
library(ModelMetrics)
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
set.seed(1)

num_features<-dim(data_PC_train)[2]-1
#LGBM_train<-as.matrix(data_PC[,1:num_features])
LGBM_train<-as.matrix(data_PC_train[,1:num_features])
LGBM_test<-as.matrix(data_PC_test[,1:num_features])

params <- list(objective="binary")
lgbm_model <- lightgbm(
  params = params
  , data = LGBM_train
  , nrounds = num_trees
  , boosting='gbdt'
  , learning_rate = 0.02
  , max_depth = 5
  , label=data_PC_train$CL_10
)
lgbm_model
model_matrix_all<-model.matrix(CL_10 ~ . - 1, data_PC)
model_martix_train <- model.matrix(CL_10 ~ . - 1, data_PC_train)
model_martix_test <- model.matrix(CL_10 ~ . - 1, data_PC_test)



lolli<-lollipop(lgbm_model,model_matrix_all)
plot(lolli,labels='topAll',log_scale=T,threshold = 0.2)

imp<-importance(lgbm_model,model_matrix_all,option='both')
plot(imp,top = 15)
inter<-interactions(lgbm_model,model_matrix_all,option='interactions')
plot(inter)
inter2<-interactions(lgbm_model,model_matrix_all,option='pairs')
plot(inter2)

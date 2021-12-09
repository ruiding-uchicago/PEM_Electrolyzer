data_PC=read.csv("datatest_PC.csv",head=T)
library(randomForest)
data_PC_train=read.csv("training_data_PC.csv",head=T)
data_PC_test=read.csv("test_data_PC.csv",head=T)
head(data_PC_train)
head(data_PC_test)
set.seed(1)
forest <- randomForest(Pt_Util ~ ., data = data_PC_train,ntree=100, importance = TRUE)
forest
plot(forest, main = "Learning curve of the forest")
PTU_predict <- predict(forest, data_PC_train)

plot(data_PC_train$Pt_Util, PTU_predict, main = 'Training Set',
     xlab = 'Real_Pt_Consumption_per_kW@0.65V (mgpt kW-1)', ylab = 'Predict_Pt_Consumption_per_kW@0.65V (mgpt kW-1)')
abline(0, 1)

#使用测试集，评估预测性能
PTU_predict <- predict(forest, data_PC_test)

plot(data_PC_test$Pt_Util, PTU_predict, main = 'Test Set',
     xlab = 'Real_Pt_Consumption_per_kW@0.65V (mgpt kW-1)', ylab = 'Predict_Pt_Consumption_per_kW@0.65V (mgpt kW-1)')
abline(0, 1)
interactions_frame <- min_depth_interactions(forest, mean_sample = "relevant_trees", uncond_mean_sample = "relevant_trees")
plot_min_depth_interactions(interactions_frame)
write.csv(interactions_frame,paste0('test/inter_sum','.csv'),row.names=FALSE,fileEncoding='UTF-8')

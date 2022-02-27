#Author: Aryan
#Date: 25-02-2022
# Load libraries
library(mlbench)
library(caret)
library(caretEnsemble)
library(dplyr)
library(naniar)
library(SHAPforxgboost)
library(xgboost)
library(data.table)
library(ggplot2)
df<-read.csv("C:\\Users\\itsme\\Documents\\neosoft.csv",na.strings = c("","NA"))
# summary(df)
# str(df)
df<-na.omit(df)
#cols<-c('seg_cabin','pax_tax','pax_fcny','recent_gap_day','dist_i_cnt_y3','seg_dep_month','dist_all_cnt_mean','flt_bag_cnt_mean','avg_dist_cnt_max','tkt_i_amt_max','pref_month_y3_1','nation_name','tkt_avg_amt_max','dist_all_cnt_max','ffp_nbr')
#df<-df[cols]
df<-df[,c('pax_tax','pax_fcny','dist_i_cnt_y3','dist_all_cnt_y3','flt_bag_cnt_y3','avg_dist_cnt_y3','tkt_i_amt_y3','pref_month_y3_1','tkt_avg_amt_y3','dist_all_cnt_y3','ffp_nbr','select_seat_cnt_y3')]
df["select_seat_cnt_y3"][df["select_seat_cnt_y3"] == "0"] <- "no"
df["select_seat_cnt_y3"][df["select_seat_cnt_y3"] != "0" & df["select_seat_cnt_y3"] != "no" ] <- "yes"
df$select_seat_cnt_y3<-as.factor(df$select_seat_cnt_y3)
str(df)
# Example of Stacking algorithms
# create submodels
seed<-7
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('lda', 'rpart', 'glm', 'knn', 'svmRadial')
set.seed(seed)
df<-head(df,10000)
# df['select_seat_cnt_y3']
models <- caretList(select_seat_cnt_y3~., data=df, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
# correlation between results
modelCor(results)
splom(results)
dotplot(results)

# stack using glm
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
set.seed(seed)
stack.glm <- caretStack(models, method="rf", metric="Accuracy", trControl=stackControl)
print(stack.glm)


X1 = as.matrix(df[,-12])
mod1 = xgboost::xgboost(
  data = X1, label = df$select_seat_cnt_y3, gamma = 0, eta = 1, 
 lambda = 0,nrounds = 1, verbose = F)
shap_values <- shap.values(xgb_model = mod1, X_train = X1)
shap_values$mean_shap_score
shap_values_cnt <- shap_values$shap_score

shap_long_cnt <- shap.prep(shap_contrib = shap_values_cnt, X_train = X1)
shap.plot.summary.wrap1(mod1, X1, top_n = 10)

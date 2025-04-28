library(mlbench)
library(purrr)

data("PimaIndiansDiabetes2")
ds <- as.data.frame(na.omit(PimaIndiansDiabetes2))
## fit a logistic regression model to obtain a parametric equation
logmodel <- glm(diabetes ~ .,
                data = ds,
                family = "binomial")
summary(logmodel)

cfs <- coefficients(logmodel) ## extract the coefficients
prednames <- variable.names(ds)[-9] ## fetch the names of predictors in a vector
prednames

sz <- 100000000 ## to be used in sampling
##sample(ds$pregnant, size = sz, replace = T)

dfdata <- map_dfc(prednames,
                  function(nm){ ## function to create a sample-with-replacement for each pred.
                    eval(parse(text = paste0("sample(ds$",nm,
                                             ", size = sz, replace = T)")))
                  }) ## map the sample-generator on to the vector of predictors
## and combine them into a dataframe

names(dfdata) <- prednames
dfdata

class(cfs[2:length(cfs)])

length(cfs)
length(prednames)
## Next, compute the logit values
pvec <- map((1:8),
            function(pnum){
              cfs[pnum+1] * eval(parse(text = paste0("dfdata$",
                                                     prednames[pnum])))
            }) %>% ## create beta[i] * x[i]
  reduce(`+`) + ## sum(beta[i] * x[i])
  cfs[1] ## add the intercept

## exponentiate the logit to obtain probability values of thee outcome variable
dfdata$outcome <- ifelse(1/(1 + exp(-(pvec))) > 0.5,
                         1, 0)

library(xgboost)
library(dplyr)

# Make sure outcome is numeric
dfdata$outcome <- as.numeric(dfdata$outcome)

# Function to train XGBoost and report results
run_xgb <- function(n) {
  cat("Running for dataset size:", n, "\n")
  
  # Sample n rows
  set.seed(123) 
  sample_data <- dfdata %>% slice_sample(n = n)
  
  # Split into train/test (80/20)
  set.seed(123)
  train_idx <- sample(1:n, size = 0.8 * n)
  train_data <- sample_data[train_idx, ]
  test_data <- sample_data[-train_idx, ]
  
  # Prepare matrices for xgboost
  dtrain <- xgb.DMatrix(data = as.matrix(train_data %>% select(-outcome)),
                        label = train_data$outcome)
  dtest <- xgb.DMatrix(data = as.matrix(test_data %>% select(-outcome)),
                       label = test_data$outcome)
  
  # Time the model training
  start_time <- Sys.time()
  model <- xgboost(data = dtrain,
                   objective = "binary:logistic",
                   nrounds = 50,
                   verbose = 0)
  end_time <- Sys.time()
  
  # Predict on test set
  preds <- predict(model, dtest)
  preds_class <- ifelse(preds > 0.5, 1, 0)
  
  # Calculate test error
  error_rate <- mean(preds_class != test_data$outcome)
  
  # Report
  cat("Method used: XGBoost\n")
  cat("Dataset size:", n, "\n")
  cat("Testing-set error rate:", round(error_rate, 4), "\n")
  cat("Time taken (seconds):", round(as.numeric(end_time - start_time, units = "secs"), 2), "\n")
  cat("---------------\n")
}

# Now run for different dataset sizes
sizes <- c(100, 1000, 10000, 100000, 1000000, 10000000)

for (sz in sizes) {
  run_xgb(sz)
}


library(caret)
library(xgboost)
library(dplyr)

# Make sure outcome is a factor (caret expects factor for classification)
dfdata$outcome <- as.factor(dfdata$outcome)

# Function to train XGBoost via caret and report results
run_xgb_caret <- function(n) {
  cat("Running for dataset size:", n, "\n")
  
  # Sample n rows
  set.seed(123)
  sample_data <- dfdata %>% slice_sample(n = n)
  
  # Split into train/test (80/20)
  set.seed(123)
  train_idx <- sample(1:n, size = 0.8 * n)
  train_data <- sample_data[train_idx, ]
  test_data <- sample_data[-train_idx, ]
  
  # Define caret training control with 5-fold CV
  trctrl <- trainControl(method = "cv", number = 5)
  
  # Time the model training
  start_time <- Sys.time()
  model <- train(outcome ~ .,
                 data = train_data,
                 method = "xgbTree",
                 trControl = trctrl,
                 verbose = 0)
  end_time <- Sys.time()
  
  # Predict on test set
  preds <- predict(model, newdata = test_data)
  
  # Calculate test error
  error_rate <- mean(preds != test_data$outcome)
  
  # Report
  cat("Method used: XGBoost via caret (5-fold CV)\n")
  cat("Dataset size:", n, "\n")
  cat("Testing-set error rate:", round(error_rate, 4), "\n")
  cat("Time taken (seconds):", round(as.numeric(end_time - start_time, units = "secs"), 2), "\n")
  cat("---------------\n")
}

# Now run for different dataset sizes
sizes <- c(100, 1000, 10000, 100000, 1000000, 10000000)

for (sz in sizes) {
  run_xgb_caret(sz)
}


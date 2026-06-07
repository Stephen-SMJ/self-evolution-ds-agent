# Tabular / prudential-life-insurance-assessment

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1825 quality usable score 28

File: `Tabular/prudential-life-insurance-assessment/rank1825_70pct/xgboost-example.R`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
library(xgboost)
train <- read_csv("../input/train.csv")
test  <- read_csv("../input/test.csv")
cat("training a XGBoost classifier\n")
clf <- xgboost(data        = data.matrix(train[,feature.names]),
               eval_metric = "merror")
write_csv(submission, "xgboost_submission.csv")
```

## 40pct rank 23 quality weak score 11

File: `Tabular/prudential-life-insurance-assessment/rank23_40pct/use-the-mlr3-package.irnb`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
{"metadata":{"kernelspec":{"name":"ir","display_name":"R","language":"R"},"language_info":{"name":"R","codemirror_mode":"r","pygments_lexer":"r","mimetype":"text/x-r-source","file_extension":".r","version":"4.4.0"},"kaggle":{"accelerator":"none","dataSources":[{"sourceId":4699,"databundleVersionId":35353,"sourceType":"competition"}],"dockerImageVersionId":30749,"isInternetEnabled":true,"language":"r","sourceType":"notebook","isGpuEnabled":false}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# Created by Giuseppe Casalicchio\nlibrary(Metrics)\nlibrary(Hmisc)\nlibrary(xgboost)\nlibrary(checkmate)\nlibrary(mlr3verse)\nlibrary(paradox)\nlibrary(data.table)\n#library(future)\n#library(parallelly)\n#availableCores(constraints = \"multicore\")\n#future::plan(\"multicore\")\n\n\n\n## Create Evaluation Function\nSQWKfun = function(x, truth, response, ...) {\n  #x = seq(1.5, 7.5, by = 1) #quantile(response, probs = c(0, cumsum(table(truth)/length(truth))))\n  cuts = c(min(response), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(response))\n  response = as.numeric(Hmisc::cut2(response, cuts))\n  err = Metrics::ScoreQuadraticWeightedKappa(response, truth, 1, 8)\n  return(err)\n}\n\n## Create Evaluation Function\nSQWKfun2 = function(truth, response, ...) {\n  x = optim(par = seq(1.5, 7.5, by = 1), fn = SQWKfun, truth = truth, response = response, control = list(fnscale = -1))\n  cat(x$par, fill = TRUE)\n  return(x$value)\n}\n\n# Custom Measure for SQWK\nMeasureRegrSQWK = R6::R6Class(\"MeasureRegrSQWK\",\n  inherit = MeasureRegr,\n  public = list(\n    fun = SQWKfun2,\n    initialize = function() {\n      super$initialize(\n        id = \"regr.sqwk\",\n        range = c(-1, 1),\n        minimize = FALSE,\n        #average = \"micro\",\n        properties = c
...[truncated]
```

## 20pct rank 527 quality usable score 40

File: `Tabular/prudential-life-insurance-assessment/rank527_20pct/testwith.py`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
import xgboost as xgb
from ml_metrics import quadratic_weighted_kappa
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
    return quadratic_weighted_kappa(yhat, y)
def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    score = scorer(data[1], data[2])
    return score
xgb_num_rounds = 10
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# Found at https://www.kaggle.com/marcellonegro/prudential-life-insurance-assessment/xgb-offset0501/run/137585/code
# convert data to xgb data structure
xgtrain = xgb.DMatrix(train.drop(columns_to_drop, axis=1), train['Response'].values)
xgtest = xgb.DMatrix(test.drop(columns_to_drop, axis=1), label=test['Response'].values)
# get the parameters for xgboost
model = xgb.train(plst, xgtrain, xgb_num_rounds, learning_rates=eta_list)
print('Train score is:', eval_wrapper(train_preds, train['Response']))
train_preds = np.clip(train_preds, -0.99, 8.99)
test_preds = np.clip(test_preds, -0.99, 8.99)
offset_train_score = eval_wrapper(np.digitize(train_preds, splits), train['Response'])
print('Offset train score is:', offset_train_score)
final_test_preds = np.round(np.clip(test_preds, 1, 8)).astype(int)
preds_out.to_csv('xgb_offset_submission.csv')
```

## 10pct rank 23 quality weak score 11

File: `Tabular/prudential-life-insurance-assessment/rank23_10pct/use-the-mlr3-package.irnb`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
{"metadata":{"kernelspec":{"name":"ir","display_name":"R","language":"R"},"language_info":{"name":"R","codemirror_mode":"r","pygments_lexer":"r","mimetype":"text/x-r-source","file_extension":".r","version":"4.4.0"},"kaggle":{"accelerator":"none","dataSources":[{"sourceId":4699,"databundleVersionId":35353,"sourceType":"competition"}],"dockerImageVersionId":30749,"isInternetEnabled":true,"language":"r","sourceType":"notebook","isGpuEnabled":false}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# Created by Giuseppe Casalicchio\nlibrary(Metrics)\nlibrary(Hmisc)\nlibrary(xgboost)\nlibrary(checkmate)\nlibrary(mlr3verse)\nlibrary(paradox)\nlibrary(data.table)\n#library(future)\n#library(parallelly)\n#availableCores(constraints = \"multicore\")\n#future::plan(\"multicore\")\n\n\n\n## Create Evaluation Function\nSQWKfun = function(x, truth, response, ...) {\n  #x = seq(1.5, 7.5, by = 1) #quantile(response, probs = c(0, cumsum(table(truth)/length(truth))))\n  cuts = c(min(response), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(response))\n  response = as.numeric(Hmisc::cut2(response, cuts))\n  err = Metrics::ScoreQuadraticWeightedKappa(response, truth, 1, 8)\n  return(err)\n}\n\n## Create Evaluation Function\nSQWKfun2 = function(truth, response, ...) {\n  x = optim(par = seq(1.5, 7.5, by = 1), fn = SQWKfun, truth = truth, response = response, control = list(fnscale = -1))\n  cat(x$par, fill = TRUE)\n  return(x$value)\n}\n\n# Custom Measure for SQWK\nMeasureRegrSQWK = R6::R6Class(\"MeasureRegrSQWK\",\n  inherit = MeasureRegr,\n  public = list(\n    fun = SQWKfun2,\n    initialize = function() {\n      super$initialize(\n        id = \"regr.sqwk\",\n        range = c(-1, 1),\n        minimize = FALSE,\n        #average = \"micro\",\n        properties = c
...[truncated]
```

## 1st rank 23 quality weak score 16

File: `Tabular/prudential-life-insurance-assessment/rank23_1st/use-the-mlr3-package.irnb`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
{"metadata":{"kernelspec":{"name":"ir","display_name":"R","language":"R"},"language_info":{"name":"R","codemirror_mode":"r","pygments_lexer":"r","mimetype":"text/x-r-source","file_extension":".r","version":"4.4.0"},"kaggle":{"accelerator":"none","dataSources":[{"sourceId":4699,"databundleVersionId":35353,"sourceType":"competition"}],"dockerImageVersionId":30749,"isInternetEnabled":true,"language":"r","sourceType":"notebook","isGpuEnabled":false}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# Created by Giuseppe Casalicchio\nlibrary(Metrics)\nlibrary(Hmisc)\nlibrary(xgboost)\nlibrary(checkmate)\nlibrary(mlr3verse)\nlibrary(paradox)\nlibrary(data.table)\n#library(future)\n#library(parallelly)\n#availableCores(constraints = \"multicore\")\n#future::plan(\"multicore\")\n\n\n\n## Create Evaluation Function\nSQWKfun = function(x, truth, response, ...) {\n  #x = seq(1.5, 7.5, by = 1) #quantile(response, probs = c(0, cumsum(table(truth)/length(truth))))\n  cuts = c(min(response), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(response))\n  response = as.numeric(Hmisc::cut2(response, cuts))\n  err = Metrics::ScoreQuadraticWeightedKappa(response, truth, 1, 8)\n  return(err)\n}\n\n## Create Evaluation Function\nSQWKfun2 = function(truth, response, ...) {\n  x = optim(par = seq(1.5, 7.5, by = 1), fn = SQWKfun, truth = truth, response = response, control = list(fnscale = -1))\n  cat(x$par, fill = TRUE)\n  return(x$value)\n}\n\n# Custom Measure for SQWK\nMeasureRegrSQWK = R6::R6Class(\"MeasureRegrSQWK\",\n  inherit = MeasureRegr,\n  public = list(\n    fun = SQWKfun2,\n    initialize = function() {\n      super$initialize(\n        id = \"regr.sqwk\",\n        range = c(-1, 1),\n        minimize = FALSE,\n        #average = \"micro\",\n        properties = c
...[truncated]
```

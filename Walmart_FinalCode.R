# ----------------------------------------------------------------------------
# 0. SETUP & LIBRARY MANAGEMENT
# ----------------------------------------------------------------------------
install_and_load <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE)) {
      message(paste("Installing missing package:", pkg))
      install.packages(pkg, dependencies = TRUE)
      library(pkg, character.only = TRUE)
    } else {
      library(pkg, character.only = TRUE)
    }
  }
}

required_packages <- c(
  "data.table", "ggplot2", "lubridate", "zoo", "patchwork", "scales",
  "ggcorrplot", "magrittr", "dplyr", "cluster", "factoextra",
  "caTools", "ranger", "ROCR", "tsibble", "fable", "fabletools",
  "fable.prophet", "feasts", "xgboost", "Metrics", "fastshap",
  "doParallel", "foreach", "grid"
)

install_and_load(required_packages)
data.table::setDTthreads(threads = 1L)

reset_graphics <- function() {
  if (!is.null(dev.list())) graphics.off()
}

# --- CUSTOM THEME:
  ggplot2::theme_minimal() +
theme_black_text <- function() {
    ggplot2::theme(
      text = ggplot2::element_text(color = "black"),
      axis.title = ggplot2::element_text(color = "black", face = "bold"),
      axis.text = ggplot2::element_text(color = "black"),
      plot.title = ggplot2::element_text(color = "black", face = "bold", size = 14),
      plot.subtitle = ggplot2::element_text(color = "black", size = 11),
      legend.text = ggplot2::element_text(color = "black"),
      legend.title = ggplot2::element_text(color = "black", face = "bold"),
      strip.text = ggplot2::element_text(color = "black", face = "bold"),
      panel.grid.major = ggplot2::element_line(color = "#e0e0e0"),
      panel.grid.minor = ggplot2::element_blank(),
      legend.position = "bottom"
    )
}
col_bar_fill  <- "#2171b5"   # Medium Blue
col_line_dark <- "#08306b"   # Dark Blue (Actuals)
col_line_light<- "#6baed6"   # Light Blue (Forecasts)
col_grad_low  <- "#deebf7"   # Lightest Blue
col_grad_high <- "#08306b"   # Darkest Blue
col_training  <- "#333333"   # Grey/Black for History

# ----------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------
CONFIG <- list(
  base_path = "C:/Users/Manveen/Downloads/DATA MINING PROJECT/WALMART/",
  k_clusters = 3,
  ts_store_id = 1,
  ts_dept_id = 1,
  ts_test_weeks = 20,
  
  class_split_ratio = 0.7,
  class_quantile_threshold = 0.75,
  
  global_test_weeks = 10,
  rf_num_trees = 500,
  
  shap_nsim = 10,
  shap_sample_size = 2000,
  shap_baseline_size = 100
)

# ----------------------------------------------------------------------------
# 2. LOAD DATA (TRAIN + FEATURES + STORES)  --> merged_dt
# ----------------------------------------------------------------------------
train_dt    <- data.table::fread(file.path(CONFIG$base_path, "train.csv"))
features_dt <- data.table::fread(file.path(CONFIG$base_path, "features.csv"))
stores_dt   <- data.table::fread(file.path(CONFIG$base_path, "stores.csv"))

merged_dt <- merge(train_dt, stores_dt, by = "Store")
merged_dt <- merge(merged_dt, features_dt, by = c("Store", "Date", "IsHoliday"))

rm(train_dt, features_dt, stores_dt); gc()

# ----------------------------------------------------------------------------
# 3. EDA & CLEANING (on merged_dt)
# ----------------------------------------------------------------------------

# Fix negative sales
merged_dt[Weekly_Sales < 0, Weekly_Sales := 0]

# Replace NA MarkDowns with 0
for (col in paste0("MarkDown", 1:5)) {
  set(merged_dt, which(is.na(merged_dt[[col]])), col, 0)
}

# Forward-fill CPI and Unemployment within each store, then fill remaining NAs with 0
merged_dt[, `:=`(
  CPI = zoo::na.locf(CPI, na.rm = FALSE),
  Unemployment = zoo::na.locf(Unemployment, na.rm = FALSE)
), by = Store]

merged_dt[is.na(CPI), CPI := 0]
merged_dt[is.na(Unemployment), Unemployment := 0]

# --- Plot 1: Histogram of Weekly_Sales ---
print("--- EDA: Weekly_Sales Distribution (Histogram) ---")
p_hist <- ggplot2::ggplot(merged_dt, ggplot2::aes(x = Weekly_Sales)) +
  ggplot2::geom_histogram(bins = 50, fill = col_bar_fill, color = "white") +
  ggplot2::scale_x_continuous(labels = scales::dollar_format()) +
  ggplot2::labs(
    title = "Frequency Distribution of Weekly Sales",
    subtitle = "Distribution across all stores and departments",
    x = "Weekly Sales",
    y = "Count"
  ) +
  theme_black_text()
print(p_hist)

# --- Plot 2: Total Weekly Sales Over Time ---
print("--- EDA: Total Weekly Sales Over Time ---")
weekly_total <- merged_dt[
  ,
  .(Weekly_Sales_Total = sum(Weekly_Sales)),
  by = Date
][order(Date)]

p_ts <- ggplot2::ggplot(weekly_total, ggplot2::aes(x = Date, y = Weekly_Sales_Total)) +
  ggplot2::geom_line(linewidth = 0.8, color = col_line_dark) +
  ggplot2::scale_y_continuous(labels = scales::dollar_format()) +
  ggplot2::labs(
    title = "Aggregate Weekly Sales Trajectory",
    subtitle = "Total sales across all Walmart stores over time",
    x = "Date",
    y = "Total Weekly Sales"
  ) +
  theme_black_text()
print(p_ts)

# --- Plot 3: Correlation Heatmap (Fixed Variables + Blue) ---
print("--- Generating Correlation Heatmap ---")

# Prepare data for numeric correlation
heatmap_dt <- copy(merged_dt)
heatmap_dt[, IsHoliday := as.numeric(as.logical(IsHoliday))]
heatmap_dt[, Type := as.numeric(as.factor(Type))]

# Variables matching your heatmap image
corr_cols <- c(
  "Weekly_Sales", "Temperature", "Fuel_Price",
  "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
  "CPI", "Unemployment", "IsHoliday", "Type", "Size"
)

corr_mat <- cor(heatmap_dt[, ..corr_cols], use = "pairwise.complete.obs")

tryCatch({
  p_corr <- ggcorrplot::ggcorrplot(
    corr_mat,
    type = "lower",
    lab = TRUE,
    title = "Correlation Matrix of Key Business Metrics",
    colors = c(col_grad_low, "white", col_grad_high), # Blue Gradient
    outline.color = "white",
    lab_col = "black",
    tl.col = "black"
  ) + theme_black_text()
  print(p_corr)
}, error = function(e) {
  print("Warning: Could not render heatmap. Skipping.")
})

# ----------------------------------------------------------------------------
# 4. FEATURE ENGINEERING (update merged_dt in place)
# ----------------------------------------------------------------------------
merged_dt[, `:=`(
  Date = as.Date(Date),
  Week = lubridate::week(Date),
  Month = lubridate::month(Date, label = TRUE),
  Year = lubridate::year(Date),
  IsHoliday = as.factor(IsHoliday),
  Type = as.factor(Type)
)]

# Special event dates
dates_sb <- c("2010-02-12", "2011-02-11", "2012-02-10")
dates_tg <- as.Date(c("2010-11-26", "2011-11-25"))
dates_xm <- as.Date(c("2010-12-31", "2011-12-30"))

merged_dt[, `:=`(
  IsSuperBowlWeek    = as.factor(Date %in% as.Date(dates_sb)),
  IsThanksgivingWeek = as.factor(Date %in% dates_tg),
  IsChristmasWeek    = as.factor(Date %in% dates_xm)
)]

pre_tg <- unlist(lapply(dates_tg, function(d) seq(d - 21, d - 1, by = "day")))
pre_xm <- unlist(lapply(dates_xm, function(d) seq(d - 21, d - 1, by = "day")))

merged_dt[, `:=`(
  Pre_Thanksgiving_Window = as.factor(Date %in% pre_tg),
  Pre_Christmas_Window    = as.factor(Date %in% pre_xm)
)]

# Lags (within each Store-Dept)
merged_dt <- merged_dt[order(Store, Dept, Date)]
merged_dt[, `:=`(
  Sales_Lag_1  = shift(Weekly_Sales, 1,  type = "lag"),
  Sales_Lag_52 = shift(Weekly_Sales, 52, type = "lag")
), by = .(Store, Dept)]

merged_dt[is.na(Sales_Lag_1),  Sales_Lag_1  := 0]
merged_dt[is.na(Sales_Lag_52), Sales_Lag_52 := 0]

# ----------------------------------------------------------------------------
# 5. CLUSTERING (Store-level, using ONLY pre-test data)
# ----------------------------------------------------------------------------

# Use only pre-test data (avoid leakage from last global_test_weeks)
cutoff_cluster <- max(merged_dt$Date) - lubridate::weeks(CONFIG$global_test_weeks)
cluster_base <- merged_dt[Date < cutoff_cluster]

store_feats <- cluster_base[
  ,
  .(
    Avg_Sales = mean(Weekly_Sales),
    Size = mean(Size)
  ),
  by = .(Store)
]

store_scaled <- scale(store_feats[, .(Avg_Sales, Size)])

# Elbow plot
print("--- Elbow Plot ---")
set.seed(123)
tryCatch({
  p_elbow <- factoextra::fviz_nbclust(
    store_scaled,
    kmeans,
    method = "wss",
    linecolor = col_line_dark
  ) +
    ggplot2::labs(title = "Optimal Cluster Selection (Elbow Method)") +
    theme_black_text()
  print(p_elbow)
}, error = function(e) {
  message("graphics error.")
})

# K-means clustering
set.seed(123)
km_res <- kmeans(store_scaled, centers = CONFIG$k_clusters, nstart = 25)
store_feats[, Cluster := as.factor(km_res$cluster)]

# Cluster plot
print("--- Cluster Plot ---")
tryCatch({
  p_cluster <- factoextra::fviz_cluster(
    km_res,
    data = store_scaled,
    geom = "point",
    main = "Store Segmentation (Sales vs. Size)",
    palette = c("#9ecae1", "#4292c6", "#08306b"), # 3 distinct blue shades
    ellipse.type = "convex",
    ggtheme = theme_black_text()
  )
  print(p_cluster)
}, error = function(e) {
  message("error")
})

# Cluster profile + merge back to merged_dt
print("--- CLUSTER PROFILE INFORMATION ---")
merged_dt <- merge(
  merged_dt,
  store_feats[, .(Store, Cluster)],
  by = "Store",
  all.x = TRUE
)

profile <- store_feats[
  ,
  .(
    Store_Count = .N,
    Avg_Sales = round(mean(Avg_Sales), 2),
    Avg_Size = round(mean(Size), 0)
  ),
  by = .(Cluster)
][order(Cluster)]

print(profile)

# ----------------------------------------------------------------------------
# 6. TIME SERIES SHOWDOWN (Single Store-Dept)
# ----------------------------------------------------------------------------

# Aggregate for chosen Store/Dept
sales_ts_dt <- merged_dt[
  Store == CONFIG$ts_store_id & Dept == CONFIG$ts_dept_id,
  .(Weekly_Sales = sum(Weekly_Sales)),
  by = Date
][order(Date)]

print("--- Time Series: Raw Store–Dept Weekly Sales ---")
p_raw_ts <- ggplot2::ggplot(sales_ts_dt, ggplot2::aes(x = Date, y = Weekly_Sales)) +
  ggplot2::geom_line(linewidth = 0.7, color = col_line_dark) +
  ggplot2::scale_y_continuous(labels = scales::dollar_format()) +
  ggplot2::labs(
    title = paste("Historical Sales Trajectory - Store", CONFIG$ts_store_id, "Dept", CONFIG$ts_dept_id),
    x = "Date",
    y = "Weekly_Sales"
  ) +
  theme_black_text()
print(p_raw_ts)

# Convert to tsibble
sales_ts <- tsibble::as_tsibble(sales_ts_dt, index = Date)

test_weeks <- CONFIG$ts_test_weeks
train_ts <- sales_ts[1:(nrow(sales_ts) - test_weeks), ]
test_ts  <- sales_ts[(nrow(sales_ts) - test_weeks + 1):nrow(sales_ts), ]

# Regression-style data with lag
ts_reg_data <- copy(sales_ts_dt)
ts_reg_data[, `:=`(
  Week = lubridate::week(Date),
  Year = lubridate::year(Date),
  Sales_Lag_52 = data.table::shift(Weekly_Sales, 52)
)]
ts_reg_data <- ts_reg_data[complete.cases(ts_reg_data)]

train_reg_ts <- ts_reg_data[Date %in% train_ts$Date]
test_reg_ts  <- ts_reg_data[Date %in% test_ts$Date]

# Fit ETS, ARIMA, Prophet
print("--- Fitting Time Series Models (ETS, ARIMA, Prophet, RF, XGB) ---")
fit_base <- train_ts %>%
  fabletools::model(
    ETS = fable::ETS(Weekly_Sales),
    ARIMA = fable::ARIMA(Weekly_Sales),
    Prophet = fable.prophet::prophet(Weekly_Sales ~ season(period = "year", order = 2))
  )

fc_base <- fabletools::forecast(fit_base, h = paste(test_weeks, "weeks"))

pred_ets <- fc_base %>%
  dplyr::filter(.model == "ETS") %>%
  dplyr::pull(.mean)
pred_arima <- fc_base %>%
  dplyr::filter(.model == "ARIMA") %>%
  dplyr::pull(.mean)
pred_prophet <- fc_base %>%
  dplyr::filter(.model == "Prophet") %>%
  dplyr::pull(.mean)

# RF (raw scale)
rf_raw <- ranger::ranger(
  formula = Weekly_Sales ~ Week + Year + Sales_Lag_52,
  data = train_reg_ts,
  num.trees = 500
)
pred_rf_raw <- predict(rf_raw, data = test_reg_ts)$predictions

# XGB (raw)
dtrain_raw <- xgboost::xgb.DMatrix(
  data = as.matrix(train_reg_ts[, .(Week, Year, Sales_Lag_52)]),
  label = train_reg_ts$Weekly_Sales
)
dtest_raw <- xgboost::xgb.DMatrix(
  data = as.matrix(test_reg_ts[, .(Week, Year, Sales_Lag_52)])
)
xgb_raw <- xgboost::xgb.train(
  params = list(objective = "reg:squarederror", eta = 0.1),
  data = dtrain_raw,
  nrounds = 500,
  verbose = 0
)
pred_xgb_raw <- predict(xgb_raw, newdata = dtest_raw)

# RF (log scale)
rf_log <- ranger::ranger(
  formula = log1p(Weekly_Sales) ~ Week + Year + Sales_Lag_52,
  data = train_reg_ts,
  num.trees = 500
)
pred_rf_log <- expm1(predict(rf_log, data = test_reg_ts)$predictions)

# XGB (log scale)
dtrain_log <- xgboost::xgb.DMatrix(
  data = as.matrix(train_reg_ts[, .(Week, Year, Sales_Lag_52)]),
  label = log1p(train_reg_ts$Weekly_Sales)
)
xgb_log <- xgboost::xgb.train(
  params = list(
    objective = "reg:squarederror",
    eta = 0.02,
    max_depth = 6
  ),
  data = dtrain_log,
  nrounds = 1000,
  verbose = 0
)
pred_xgb_log <- expm1(predict(xgb_log, newdata = dtest_raw))

# Grid search for ensembles of RF(Log), XGB(Log), Prophet
print("--- Running Grid Search for Log-RF + Log-XGB + Prophet Ensembles ---")
patterns <- list(
  c(35, 35, 30), c(40, 30, 30), c(35, 40, 25), c(40, 35, 25), c(30, 40, 30),
  c(25, 50, 25), c(50, 25, 25), c(35, 45, 20), c(45, 35, 20), c(30, 50, 20),
  c(50, 30, 20), c(20, 60, 20), c(60, 20, 20), c(33, 33, 33)
)

calc_rmse <- function(p) Metrics::rmse(test_reg_ts$Weekly_Sales, p)
calc_r2_ts <- function(act, pred) 1 - (sum((act - pred)^2) / sum((act - mean(act))^2))
calc_mae <- function(p) Metrics::mae(test_reg_ts$Weekly_Sales, p)

results_list <- list()
for (i in seq_along(patterns)) {
  w <- patterns[[i]] / 100
  pred_ens <- (w[1] * pred_rf_log) + (w[2] * pred_xgb_log) + (w[3] * pred_prophet)
  name <- paste0("ENS: ", patterns[[i]][1], "/", patterns[[i]][2], "/", patterns[[i]][3])
  results_list[[i]] <- data.table(
    Model = name,
    RMSE = calc_rmse(pred_ens),
    MAE = calc_mae(pred_ens),
    R2 = calc_r2_ts(test_reg_ts$Weekly_Sales, pred_ens)
  )
}

singles <- list(
  list("Base: RF (Log)", pred_rf_log),
  list("Base: XGB (Log)", pred_xgb_log),
  list("Base: RF (Raw)", pred_rf_raw),
  list("Base: XGB (Raw)", pred_xgb_raw),
  list("Base: Prophet", pred_prophet),
  list("Base: ARIMA", pred_arima),
  list("Base: ETS", pred_ets)
)

single_results <- lapply(singles, function(x) {
  data.table(
    Model = x[[1]],
    RMSE = calc_rmse(x[[2]]),
    MAE = calc_mae(x[[2]]),
    R2 = calc_r2_ts(test_reg_ts$Weekly_Sales, x[[2]])
  )
})

all_results_ts <- rbindlist(c(results_list, single_results))
print("--- FINAL TIME SERIES LEADERBOARD (RMSE + R2) ---")
print(all_results_ts[order(RMSE)])

# Choose winner by RMSE
best_model_name <- all_results_ts[order(RMSE)][1, Model]
if (grepl("ENS", best_model_name)) {
  parts <- as.numeric(unlist(strsplit(gsub("ENS: ", "", best_model_name), "/")))
  pred_winner <- (parts[1] / 100 * pred_rf_log) +
    (parts[2] / 100 * pred_xgb_log) +
    (parts[3] / 100 * pred_prophet)
} else {
  pred_winner <- pred_rf_log
}

# Winner vs Actual
print("--- Time Series: Winner vs Actual (Zoomed View) ---")
plot_dt <- data.table(Date = test_reg_ts$Date, Forecast = pred_winner)
p_verdict <- ggplot2::ggplot() +
  ggplot2::geom_line(
    data = train_reg_ts[Date > as.Date("2011-12-31")],
    ggplot2::aes(x = Date, y = Weekly_Sales, color = "Training"),
    linewidth = 0.7
  ) +
  ggplot2::geom_line(
    data = test_reg_ts,
    ggplot2::aes(x = Date, y = Weekly_Sales, color = "Actual"),
    linewidth = 0.8
  ) +
  ggplot2::geom_line(
    data = plot_dt,
    ggplot2::aes(x = Date, y = Forecast, color = "Winner"),
    linetype = "dashed",
    linewidth = 0.9
  ) +
  ggplot2::scale_color_manual(values = c("Training" = col_training, "Actual" = col_line_dark, "Winner" = col_line_light)) +
  ggplot2::labs(
    title = "Final Time Series Verdict (Zoomed)",
    subtitle = paste("Winning Model:", best_model_name)
  ) +
  theme_black_text()
print(p_verdict)

# Individual model forecast plots
print("--- Time Series: Individual Model Forecast Charts ---")
models_to_plot <- list(
  "ARIMA" = pred_arima,
  "ETS" = pred_ets,
  "Prophet" = pred_prophet,
  "RF(Log)" = pred_rf_log,
  "XGB(Log)" = pred_xgb_log,
  "Winner" = pred_winner
)
actuals_ts <- data.table(Date = test_reg_ts$Date, Actual = test_reg_ts$Weekly_Sales)

for (m in names(models_to_plot)) {
  dt_plot <- copy(actuals_ts)
  dt_plot[, Forecast := models_to_plot[[m]]]
  p_ts_model <- ggplot2::ggplot(dt_plot, ggplot2::aes(x = Date)) +
    ggplot2::geom_line(ggplot2::aes(y = Actual, color = "Actual")) +
    ggplot2::geom_line(
      ggplot2::aes(y = Forecast, color = "Forecast"),
      linetype = "dashed"
    ) +
    ggplot2::scale_color_manual(values = c("Actual" = col_line_dark, "Forecast" = col_line_light)) +
    ggplot2::labs(title = paste("Forecast:", m)) +
    theme_black_text()
  print(p_ts_model)
}

# ----------------------------------------------------------------------------
# 7. CLASSIFICATION: HIGH vs LOW SALES WEEKS
# ----------------------------------------------------------------------------
class_dt <- merged_dt[
  ,
  .(
    Weekly_Sales = sum(Weekly_Sales),
    Temp = mean(Temperature),
    Fuel = mean(Fuel_Price),
    MkDn = sum(MarkDown1 + MarkDown2 + MarkDown3 + MarkDown4 + MarkDown5),
    CPI = mean(CPI),
    Unemp = mean(Unemployment),
    IsHoliday = max(as.numeric(IsHoliday))
  ),
  by = .(Store, Date, Week, Year)
]

feats_class <- c("Temp", "Fuel", "MkDn", "CPI", "Unemp", "Week", "Year", "IsHoliday")
# Train-test split (by rows, for simplicity)
split_flag <- caTools::sample.split(class_dt$Weekly_Sales, SplitRatio = CONFIG$class_split_ratio)
train_class <- class_dt[split_flag == TRUE]
test_class  <- class_dt[split_flag == FALSE]

# Compute store-specific thresholds on TRAIN only
threshold_dt <- train_class[
  ,
  .(Threshold = quantile(Weekly_Sales, CONFIG$class_quantile_threshold)),
  by = Store
]

train_class <- merge(train_class, threshold_dt, by = "Store")
test_class  <- merge(test_class, threshold_dt, by = "Store", all.x = TRUE)

# Drop test rows where threshold missing (stores not in train)
test_class <- test_class[!is.na(Threshold)]

# Create labels
train_class[, High_Sales := as.factor(ifelse(Weekly_Sales > Threshold, "Yes", "No"))]
test_class[,  High_Sales := as.factor(ifelse(Weekly_Sales > Threshold, "Yes", "No"))]

# Complete cases
train_class <- train_class[
  complete.cases(train_class[, c(feats_class, "High_Sales"), with = FALSE])
]
test_class <- test_class[
  complete.cases(test_class[, c(feats_class, "High_Sales"), with = FALSE])
]

# Weighted RF (handle class imbalance)
counts <- table(train_class$High_Sales)
w_yes <- counts["No"] / counts["Yes"]
case_weights <- ifelse(train_class$High_Sales == "Yes", w_yes, 1)

rf_model_class <- ranger::ranger(
  formula = High_Sales ~ .,
  data = train_class[, c(feats_class, "High_Sales"), with = FALSE],
  case.weights = case_weights,
  probability = TRUE,
  num.trees = 500
)
prob_rf <- predict(rf_model_class, data = test_class)$predictions[, "Yes"]

# Logistic regression
log_model_class <- glm(
  High_Sales ~ .,
  data = train_class[, c(feats_class, "High_Sales"), with = FALSE],
  family = "binomial"
)
prob_log <- predict(log_model_class, newdata = test_class, type = "response")

# ROC + AUC
pred_rf_class  <- ROCR::prediction(prob_rf, test_class$High_Sales)
pred_log_class <- ROCR::prediction(prob_log, test_class$High_Sales)

perf_rf  <- ROCR::performance(pred_rf_class, "tpr", "fpr")
perf_log <- ROCR::performance(pred_log_class, "tpr", "fpr")

auc_rf  <- round(ROCR::performance(pred_rf_class, "auc")@y.values[[1]], 3)
auc_log <- round(ROCR::performance(pred_log_class, "auc")@y.values[[1]], 3)

df_roc_plot <- rbind(
  data.table(
    FPR = perf_rf@x.values[[1]],
    TPR = perf_rf@y.values[[1]],
    Model = "Weighted RF (Optimized)"
  ),
  data.table(
    FPR = perf_log@x.values[[1]],
    TPR = perf_log@y.values[[1]],
    Model = "Logistic Regression"
  )
)

print("--- Classification ROC Curves ---")
p_roc <- ggplot2::ggplot(df_roc_plot, ggplot2::aes(x = FPR, y = TPR, color = Model)) +
  ggplot2::geom_line(linewidth = 1.2) +
  ggplot2::geom_abline(linetype = "dashed") +
  ggplot2::scale_color_manual(values = c("Weighted RF (Optimized)" = col_line_dark, "Logistic Regression" = col_line_light)) +
  ggplot2::labs(
    title = "Classification ROC: Weighted RF vs Logistic",
    subtitle = paste("RF AUC:", auc_rf, "| LogReg AUC:", auc_log)
  ) +
  theme_black_text() +
  ggplot2::theme(legend.position = "bottom")
print(p_roc)

roc_comparison <- data.table(
  Model = c("Weighted RF (Optimized)", "Logistic Regression"),
  AUC    = c(auc_rf, auc_log)
)
print("--- ROC AUC COMPARISON TABLE ---")
print(roc_comparison)

# ----------------------------------------------------------------------------
# 8. GLOBAL REGRESSION (Hybrid RF + XGB with RMSE-weighted Ensemble)
# ----------------------------------------------------------------------------
reg_dt <- copy(merged_dt)

feats_reg <- c(
  "Store", "Dept", "IsHoliday", "Type", "Size",
  "Temperature", "Fuel_Price",
  "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
  "CPI", "Unemployment",
  "Week", "Month", "Year", "Cluster",
  "Sales_Lag_1", "Sales_Lag_52"
)

reg_dt_num <- copy(reg_dt)
for (cname in c("IsHoliday", "Type", "Month", "Cluster")) {
  reg_dt_num[, (cname) := as.numeric(get(cname))]
}

cutoff_reg <- max(reg_dt_num$Date) - lubridate::weeks(CONFIG$global_test_weeks)
train_reg <- reg_dt_num[Date < cutoff_reg]
test_reg  <- reg_dt_num[Date >= cutoff_reg]

train_reg <- train_reg[complete.cases(train_reg[, c(feats_reg, "Weekly_Sales"), with = FALSE])]
test_reg  <- test_reg[complete.cases(test_reg[, c(feats_reg, "Weekly_Sales"), with = FALSE])]

# Baseline RF
rf_base <- ranger::ranger(
  Weekly_Sales ~ .,
  data = train_reg[, c(feats_reg, "Weekly_Sales"), with = FALSE],
  num.trees = 100,
  importance = "impurity" 
)
pred_rf_base <- predict(rf_base, data = test_reg)$predictions

# XGB (log scale)
dtrain_reg_log <- xgboost::xgb.DMatrix(
  data = as.matrix(train_reg[, ..feats_reg]),
  label = log1p(train_reg$Weekly_Sales)
)
dtest_reg <- xgboost::xgb.DMatrix(
  data = as.matrix(test_reg[, ..feats_reg])
)

xgb_opt <- xgboost::xgb.train(
  params = list(
    eta = 0.02,
    max_depth = 10,
    subsample = 0.8,
    objective = "reg:squarederror"
  ),
  data = dtrain_reg_log,
  nrounds = 1000,
  verbose = 0
)
pred_xgb_opt <- expm1(predict(xgb_opt, newdata = dtest_reg))

calc_rmse_reg <- function(p) Metrics::rmse(test_reg$Weekly_Sales, p)
calc_r2_reg <- function(act, pred) 1 - (sum((act - pred)^2) / sum((act - mean(act))^2))

rmse_rf_base <- calc_rmse_reg(pred_rf_base)
rmse_xgb_opt <- calc_rmse_reg(pred_xgb_opt)
r2_rf_base   <- calc_r2_reg(test_reg$Weekly_Sales, pred_rf_base)
r2_xgb_opt   <- calc_r2_reg(test_reg$Weekly_Sales, pred_xgb_opt)

# RMSE-weighted ensemble
w_rf  <- 1 / rmse_rf_base
w_xgb <- 1 / rmse_xgb_opt
w_sum <- w_rf + w_xgb

w_rf_norm  <- w_rf  / w_sum
w_xgb_norm <- w_xgb / w_sum

pred_hybrid <- (w_rf_norm * pred_rf_base) + (w_xgb_norm * pred_xgb_opt)

rmse_hybrid <- calc_rmse_reg(pred_hybrid)
r2_hybrid   <- calc_r2_reg(test_reg$Weekly_Sales, pred_hybrid)

res_reg <- data.table(
  Model = c("Baseline RF", "Optimized XGB (Log)", "Hybrid (RMSE-weighted RF + XGB)"),
  RMSE  = c(rmse_rf_base, rmse_xgb_opt, rmse_hybrid),
  R2    = c(r2_rf_base,   r2_xgb_opt,   r2_hybrid)
)

print("--- GLOBAL REGRESSION SHOWDOWN (RMSE + R2) ---")
print(res_reg[order(RMSE)])

# Plot global hybrid forecast vs actual (aggregated by Date)
plot_reg_dt <- data.table(
  Date = test_reg$Date,
  Actual = test_reg$Weekly_Sales,
  Forecast = pred_hybrid
)
plot_reg_agg <- plot_reg_dt[
  ,
  .(Actual = sum(Actual), Forecast = sum(Forecast)),
  by = Date
]

print("--- Global Hybrid Forecast vs Actual ---")
p_global <- ggplot2::ggplot(plot_reg_agg, ggplot2::aes(x = Date)) +
  ggplot2::geom_line(
    ggplot2::aes(y = Actual, color = "Actual Data (Test Set)"),
    linewidth = 0.8
  ) +
  ggplot2::geom_line(
    ggplot2::aes(y = Forecast, color = "Ensemble Forecast"),
    linetype = "dashed",
    linewidth = 0.9
  ) +
  ggplot2::scale_color_manual(values = c("Actual Data (Test Set)" = col_line_dark, "Ensemble Forecast" = col_line_light)) +
  ggplot2::scale_y_continuous(labels = scales::dollar_format()) +
  ggplot2::labs(
    title = "Global Prediction (Hybrid Ensemble) - 2012 Zoom"
  ) +
  theme_black_text()
print(p_global)

# ----------------------------------------------------------------------------
# 9. EXPLAINABILITY: RF Feature Importance + XGBoost SHAP
# ----------------------------------------------------------------------------
print("STEP 9: Explainability (Feature Importance + SHAP)...")

# RF feature importance
if (!is.null(rf_base$variable.importance)) {
  imp_rf <- ranger::importance(rf_base)
  dt_imp_rf <- data.table(
    Feature = names(imp_rf),
    Importance = imp_rf
  )[order(-Importance)][1:15]
  
  print("--- Random Forest Feature Importance ---")
  p_rf_imp <- ggplot2::ggplot(dt_imp_rf, ggplot2::aes(x = reorder(Feature, Importance), y = Importance, fill = Importance)) +
    ggplot2::geom_col() +
    ggplot2::coord_flip() +
    ggplot2::scale_fill_gradient(low = col_grad_low, high = col_grad_high) +
    ggplot2::labs(title = "RF Feature Importance") +
    theme_black_text() + ggplot2::theme(legend.position = "none")
  print(p_rf_imp)
}

# SHAP for XGBoost (sequential, sampled)
set.seed(123)

train_data_for_shap <- train_reg
features_for_shap <- feats_reg

idx_shap <- sample(
  1:nrow(train_data_for_shap),
  min(CONFIG$shap_sample_size, nrow(train_data_for_shap))
)
X_samp <- as.matrix(train_data_for_shap[idx_shap, ..features_for_shap])

base_idx <- sample(
  1:nrow(train_data_for_shap),
  min(CONFIG$shap_baseline_size, nrow(train_data_for_shap))
)
X_base <- as.matrix(train_data_for_shap[base_idx, ..features_for_shap])

pfun_xgb <- function(object, newdata) {
  predict(object, newdata = as.matrix(newdata))
}

shap_vals <- fastshap::explain(
  xgb_opt,
  X = X_samp,
  pred_wrapper = pfun_xgb,
  nsim = CONFIG$shap_nsim,
  baseline = X_base,
  parallel = FALSE
)

shap_imp <- data.table(
  Feature = features_for_shap,
  SHAP = colMeans(abs(shap_vals))
)[order(-SHAP)][1:15]

print("--- XGBoost SHAP Importance ---")
p_shap <- ggplot2::ggplot(shap_imp, ggplot2::aes(x = reorder(Feature, SHAP), y = SHAP, fill = SHAP)) +
  ggplot2::geom_col() +
  ggplot2::coord_flip() +
  ggplot2::scale_fill_gradient(low = col_grad_low, high = col_grad_high) +
  ggplot2::labs(title = "XGBoost SHAP Importance") +
  theme_black_text() + ggplot2::theme(legend.position = "none")
print(p_shap)

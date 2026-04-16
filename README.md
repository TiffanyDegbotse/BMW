# BMW Inventory Optimizer

A machine learning pipeline that predicts days-on-lot for Priority 3 & 4 BMW vehicles and surfaces the fastest-selling color, upholstery, and wheel combinations for each model. Given historical BMW customer order data, the optimizer ranks every observed color × upholstery × wheel combination by predicted days-to-sale, answering the question: which spec sells fastest in the current market?

## Models

Four models are trained and compared. LightGBM is the best performer and uses native categorical support without one-hot encoding. XGBoost uses one-hot encoded gradient boosted trees. Ridge Regression is a linear baseline with RidgeCV automatic alpha selection across [0.1, 1, 5, 10, 50, 100, 500]. The Neural Network (TabularMLP) is a custom PyTorch MLP with entity embeddings for categorical features, hyperparameter tuned with Optuna over 30 trials.

## Features

The model uses 19 input features. Categorical features are short_description, model_group_code, ag_model_code, na_model_code, color_code, upholstery_code, wheel_code, hybrid_flag, drive_config, dealer_number, market, and region. Numeric features are engine_capacity_ltr, model_year, power_rating_kw, prod_priority_num, distrib_priority_num, retl_base_price, number_cylinders, arrival_month, and arrival_quarter.

## Data

Training data is scoped to Priority 3 and Priority 4 customer orders only. These represent vehicles built to a specific customer spec and have the clearest signal between configuration and sale speed. The target variable is days_on_lot capped at 90 days. Data is split 70/15/15 into train, validation, and test sets, stratified by days_on_lot quartile to preserve the distribution of fast and slow sellers across all splits. Models are not valid for other priority types.

## Optimizer

For each BMW model with at least 10 historical records, the optimizer identifies all color, upholstery, and wheel combinations observed at least 3 times in history, builds a synthetic vehicle for each using the median feature profile of that model with the current month and quarter substituted in for arrival timing, runs each through the trained model, and ranks them from fastest-selling to slowest. All predictions are pre-computed and stored in combo_cache.pkl to keep API response time under 100ms. Each combination is assigned a confidence score combining historical sample size (up to 60 points for 15 or more observations) and IQR consistency (up to 40 points for tight distributions). Scores at or above 60% are high confidence, 30 to 59% are medium, and below 30% are low.

## Output

The pipeline produces a ranked combo table per model from fastest to slowest, a predicted vs historical median bar chart, a color × upholstery heatmap of predicted days on lot, SHAP feature importance from the neural network, and a model comparison table and chart showing MAE, RMSE, and R² across all four models.

## Setup

pip install optuna shap torch torchvision xgboost lightgbm scikit-learn pandas numpy matplotlib seaborn joblib

Place the two Excel data files in the inventorydata/ subdirectory then run all notebook sections in order. The full pipeline including Optuna search takes approximately 30 to 60 minutes on CPU and is faster with a GPU which is auto-detected.

## Project Structure

BMW/ contains the BMW_Optimizer directory which holds the inventorydata folder for raw Excel files, lgb_model.lgb for the trained LightGBM model, ridge_model.pkl for the trained Ridge pipeline, xgb_model.pkl for the trained XGBoost model, model.pt for the neural network checkpoint, combo_cache.pkl for pre-computed combo rankings, and the optimizer notebook.

## Notes

Three inconsistent model name variants are normalised before training: X3 30 and X3 30 xDr are both renamed to X3 xDr30i, and X3 M50 xDr is renamed to X3 M50. Wheel codes are joined from the option data file using chassis number as the key. Vehicles with no matching wheel code are labelled as Standard / Not Specified. The repo does not commit raw data files.

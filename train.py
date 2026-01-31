import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor

mlflow.set_tracking_uri("http://localhost:5555")
mlflow.set_experiment("RealEstatePriceNLP")

with mlflow.start_run():
    model = XGBRegressor()
    model.fit(X_train, y_train)

    mlflow.log_param("model", "XGBRegressor")
    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(model, "model")

with mlflow.start_run():
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    mlflow.log_param("model", "XGBRegressor")
    mlflow.log_metric("rmse", rmse)

    # ✅ SAVE TFIDF
    joblib.dump(tfidf, "tfidf.pkl")
    mlflow.log_artifact("tfidf.pkl")

    # ✅ REGISTER MODEL (THIS CREATES REGISTRY ENTRY)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="RealEstatePriceModel"
    )

    print(f"✅ Training complete | RMSE: {rmse:.2f}")

from src import feature_importance

if __name__ == "__main__":
    # Classification Implementation
    f_imp_clf = FeatureImportanceClassification(
        num_samples_synthetic=1000,
        plot_importance=False,
        generate_synthetic_data=True,
        estimators=(
            ("RandomForestClassifier", RandomForestClassifier()),
            ("LGBMClassifier", LGBMClassifier()),
        ),
    )

    f_imp_clf.fit_transform()

    # Regression Implementation
    f_imp_reg = FeatureImportanceRegression(
        num_samples_synthetic=1000,
        plot_importance=False,
        generate_synthetic_data=True,
        objective="regression",
        estimators=(
            ("LGBMRegressor", LGBMRegressor()),
            ("LinearRegression", LinearRegression()),
        ),
    )
    f_imp_reg.fit_transform()

    print(f_imp_clf.feature_importance_df, "\n")
    print(f_imp_reg.feature_importance_df)

📘 Week 2 Reflection — Trees, Ensembles & Boosting
✅ What I Learned

Linear Regression (regression basics)

Works for simple, linear trends.

Coefficients explain direction & strength.

But underfits on complex, noisy data (like California Housing).

Polynomial Regression & Overfitting

More flexible models capture more detail.

But too much complexity → overfitting (memorizing noise).

Regularization (Ridge & Lasso)

Ridge shrinks coefficients, Lasso can eliminate features.

Useful to control complexity in linear models.

Logistic Regression (classification refresher)

Outputs probabilities via sigmoid.

Decision boundaries are always linear.

Great for linearly separable data (Setosa), weak otherwise (Versicolor vs Virginica).

Decision Trees

Non-linear splits create boxy decision regions.

Powerful, but prone to overfitting if too deep.

Random Forests

Many trees averaged → more stable, less variance.

Performs better than a single tree.

Gradient Boosting

Trees built sequentially, correcting each other’s errors.

Often beats Random Forest, but sensitive to hyperparameters.

🏗️ Mini-Project: Model Playground

Built a Streamlit app to compare Linear Regression, Random Forest, and Gradient Boosting on the California Housing dataset.

Features:

Train & evaluate models (R², RMSE).

Inspect feature importances / coefficients.

Side-by-side model comparison.

Custom predictions.

Learning curves to diagnose under/overfitting.

Biggest insight: Median Income is by far the strongest predictor of house prices.

💡 Key Insights

Linear models are interpretable but weak on non-linear data.

Tree ensembles (RF, GB) dominate structured/tabular datasets.

Boosting can edge out Random Forest but requires careful tuning.

Evaluation isn’t just about accuracy — you must watch for bias, variance, and overfitting.

🔮 What’s Next (Week 3)

Learn cross-validation to evaluate models more reliably.

Explore hyperparameter tuning (GridSearch, RandomizedSearch).

Dive deeper into learning & validation curves.

Understand the bias–variance tradeoff with real experiments.

Build pipelines that combine preprocessing + models seamlessly.

Mini-project: a properly validated, tuned workflow on a real dataset.
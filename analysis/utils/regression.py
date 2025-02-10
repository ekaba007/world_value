import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

# Regression config
REF_POL = 5
REF_HAPPINESS = 2
COUNTRIES_ISO = ['AUS','CAN','DEU','NLD','USA']
REGRESSORS = [# corruption
              "group_corruption",
              
              # migration
              "is_immigrant",
              "mother_immigrant",
              "father_immigrant",
              "migration_positive",
              "migration_negative",
              
              # political preference
              *[f"pol_value_{i}" for i in range(1,11) if i != REF_POL],

              # happiness dummies
              *[f"happ__{i}" for i in range(1,5) if i != REF_HAPPINESS],

              # hardships and security
              "hardships_questions",
              "security_neighborhood",
              "security_financial",
              "security_war",
              
              # demographics (including happiness)
              "age",
              "gender",
              "above_avg_inc"]

def plot(ols_model_robust, name):
  # Residuals from the model
  residuals = ols_model_robust.resid

  # Predicted values from the model
  fitted_values = ols_model_robust.fittedvalues

  # Create a figure with 3 vertical subplots; use A4 dimensions (in inches)
  fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8.27, 11.69))

  # 1. Residuals vs. Fitted Plot
  sns.residplot(x=fitted_values, y=residuals, lowess=True,
                line_kws={'color': 'red', 'lw': 1}, ax=axes[0])
  axes[0].axhline(0, linestyle='--', color='gray', linewidth=1)
  axes[0].set_title('Residuals vs Fitted')
  axes[0].set_xlabel('Fitted Values')
  axes[0].set_ylabel('Residuals')

  # 2. Q-Q Plot
  # Note: statsmodels.qqplot supports passing an "ax" parameter in recent versions.
  sm.qqplot(residuals, line='45', fit=True, ax=axes[1])
  axes[1].set_title('Q-Q Plot of Residuals')

  # 3. Histogram of Residuals
  sns.histplot(residuals, color='blue', ax=axes[2])
  axes[2].set_title('Histogram of Residuals')
  axes[2].set_xlabel('Residuals')
  axes[2].set_ylabel('Frequency')

  # Adjust layout so that subplots do not overlap
  plt.tight_layout()
  plt.savefig(f"./plots/{name}")

def vif(X_const):
    vif = pd.DataFrame()
    vif.index = X_const.columns
    vif["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    return vif

# get coefficients
def append_star(row):
  p_val = row["p-Value"]
  est = row["Estimate"]
  if p_val < 0.001:
    return f"{est}***"
  if p_val < 0.01:
    return f"{est}**"
  if p_val < 0.05:
    return f"{est}*"
  return est 

def apa_latex_est_tablular(lm, X):
  """
  Make sure to correct placement inside of the columns to {cc..} and add an appropriate label
  """
  # first create a dataframe with all the entries
  results_table = pd.DataFrame({"Estimate": round(lm.params,3).astype("str"),
                              r"$\hat\sigma_{\hat\beta}$": round(lm.HC0_se, 3).astype("str"),
                              r"$T_{H_0:\beta=0}$": round(lm.tvalues, 3).astype("str"),
                              "p-Value": round(lm.pvalues, 3),
                              "VIF": round(vif(X)["VIF"], 3).astype("str")})
  
  # then append significant stars
  results_table["Estimate"] = results_table.apply(lambda row: append_star(row), axis=1)
  # then remove leading zero from p-values
  results_table["p-Value"] = results_table["p-Value"].astype("str").str.lstrip("0")
  results_table["VIF"] = results_table["VIF"].str.lstrip("0")

  results_table.loc[results_table["p-Value"] == ".0", ["p-Value"]] = "<.001"
  results_table.loc[results_table[r"$\hat\sigma_{\hat\beta}$"] == "0.0", [r"$\hat\sigma_{\hat\beta}$"]] = "<0.001"
  
  # make features formatted columns formatted correctly
  results_table.index = [r"\texttt {" + key.replace("_", r"\_") + "}" for key in results_table.index]

  print(results_table.to_latex())


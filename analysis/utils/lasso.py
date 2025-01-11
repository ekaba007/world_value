import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt


def plot_lasso_coef(df: pd.DataFrame, dependent_variable: str, alpha: float, spare_vars = [], country_code = 0) -> None:
  """
  Plot lasso coefficients

  Parameters:
    df(pd.DataFrame): imported DataFrame from wave7.csv
    dependent_variable(str): dependent variable as a string
    alpha(float): scaling of the penalty term. Higher -> less features, Lower -> more features
    spare_vars(list of str): Additional features you want include
    country_code(int): code from column B_COUNTRY

  Returns: None  
  """
  # Filtering the data for country code and build training data
  if country_code != 0:
    qf = df.loc[df.B_COUNTRY == country_code, [*df.columns[df.columns.str.match(r"Q\d+")], *spare_vars, dependent_variable]]
    trust_columns = [f"Q{i}" for i in range(53, 90)]
    X, y = qf.drop(columns=trust_columns).drop(columns=[dependent_variable]).drop(columns=["Q82_EU"]), qf[dependent_variable]
  else:
    qf = df.loc[:, [*df.columns[df.columns.str.match(r"Q\d+")], *spare_vars, dependent_variable]]
    trust_columns = [f"Q{i}" for i in range(53, 90)]
    X, y = qf.drop(columns=trust_columns).drop(columns=[dependent_variable]).drop(columns=["Q82_EU"]), qf[dependent_variable]  
  
  # fitting the lasso regression
  las = Lasso(alpha)
  las.fit(X,y)

  relevant_index = np.where(las.coef_ != 0)[0]
  relevant_questions = X.columns[relevant_index]
  lasso_res =  pd.DataFrame({"question": relevant_questions,
                            "coef": las.coef_[relevant_index]})
  # Sort your data by coefficient (so that X and y align in the same order)
  sorted_data = lasso_res.sort_values(by="coef", ascending=True)

  questions_sorted = sorted_data["question"]
  coef_sorted = sorted_data["coef"]

  # Create a colormap and normalize the coefficient values
  norm = plt.Normalize(coef_sorted.min(), coef_sorted.max())
  cmap = plt.cm.RdYlGn  # red -> yellow -> green

  # Map each coefficient to a color
  colors = cmap(norm(coef_sorted))

  plt.figure(figsize=(10, 8))
  plt.bar(questions_sorted, coef_sorted, color=colors)
  plt.xticks(rotation=60, ha='right')
  plt.title(f"Lasso ceofficients for $\\alpha = {alpha}$ and country code {country_code}" if country_code else f"Lasso ceofficients for $\\alpha = {alpha}$")
  plt.gca().set_facecolor("lightgray")
  plt.tight_layout()
  plt.grid(True, color="white")
  plt.show()
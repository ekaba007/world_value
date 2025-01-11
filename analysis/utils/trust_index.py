import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def attach_distrust_index(df: pd.DataFrame) -> pd.DataFrame:
  """
  # Creating a trust index

  Attaches trust indeces for base, national and international trust to the dataframe.
  Furthermore, all trust related questions are preprocessed by imputing with the median and MinMaxScaled.

  Parameters:
    df(pd.DataFrame): the wave7.csv dataframe stored by the data_filter_merge.ipynb
  
  Returns:
    pd.DataFrame: Processed dataframe with the distrust indeces in `base_distrust_index`, `national_distrust_index`, `international_distrust_index`
  """
  trust_questions = [f"Q{i}" for i in range(53, 90)]
  base_trust_qustions = [f"Q{i}" for i in range(53, 64)]
  national_trust_questions  = [f"Q{i}" for i in range(64, 82)]
  international_trust_questions = [f"Q{i}" for i in range(82, 90)]

  result = df.copy()

  # 1. imputing with median --------------------------------------------
  # create a median dict with all the relevant values
  median_dict = {}
  countries = result.B_COUNTRY.unique()

  for ct in countries:
      median_dict[ct] = {}

  for tq in trust_questions:
      for ct in countries:
          median_dict[ct][tq] = result.loc[(result[tq] > 0) & (result.B_COUNTRY == ct), tq].median()
  
  # now populate the dataframe with the median values
  for tq in trust_questions:
      result[tq] = result.loc[:, [tq, "B_COUNTRY"]].apply(lambda row: median_dict[row["B_COUNTRY"]][tq] if row[tq] <= 0 else row[tq], axis=1)
    
  # 2. Minmax scaling --------------------------------------------
  scaler = MinMaxScaler()
  scaler.fit(result.loc[:, trust_questions])
  result.loc[:, trust_questions] = scaler.transform(result.loc[:, trust_questions])

  # 3. Creating trust indeces by rowwise mean --------------------------------------------
  result["base_distrust_index"] = result.loc[:, base_trust_qustions].mean(axis=1)
  result["national_distrust_index"] = result.loc[:, national_trust_questions].mean(axis=1)
  result["international_distrust_index"] = result.loc[:, international_trust_questions].mean(axis=1)

  return result
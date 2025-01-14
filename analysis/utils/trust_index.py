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

def attach_migration_index(df: pd.DataFrame) -> pd.DataFrame:
  """
  # Creating a migration perception

  Attaches migration perception variables to the dataframe.
  Furthermore, all migration related questions are preprocessed by imputing with the median and MinMaxScaled.

  Parameters:
    df(pd.DataFrame): the wave7.csv dataframe stored by the data_filter_merge.ipynb
  
  Returns:
    pd.DataFrame: Processed dataframe with the migration perception variables ['migration_perception'],['migration_positive'],['migration_negative']]
  """

  migration_questions = [f"Q{i}" for i in range(121,130)]
  migration_perception = ["Q121"]
  migration_positive = ["Q122","Q123","Q125","Q127"]
  migration_negative =["Q124","Q126","Q128","Q129"]

  result = df.copy()

  # 1. imputing with median --------------------------------------------
  # create a median dict with all the relevant values
  median_dict = {}
  countries = result.B_COUNTRY.unique()

  for ct in countries:
      median_dict[ct] = {}

  for tq in migration_questions:
      for ct in countries:
          median_dict[ct][tq] = result.loc[(result[tq] > 0) & (result.B_COUNTRY == ct), tq].median()
  
  # now populate the dataframe with the median values
  for tq in migration_questions:
      result[tq] = result.loc[:, [tq, "B_COUNTRY"]].apply(lambda row: median_dict[row["B_COUNTRY"]][tq] if row[tq] <= 0 else row[tq], axis=1)
    
  # 2. Minmax scaling --------------------------------------------
  scaler = MinMaxScaler()
  scaler.fit(result.loc[:, migration_perception])
  result.loc[:, migration_perception] = scaler.transform(result.loc[:, migration_perception])

  scaler.fit(result.loc[:, migration_positive])
  result.loc[:, migration_positive] = scaler.transform(result.loc[:, migration_positive])


  scaler.fit(result.loc[:, migration_negative])
  result.loc[:, migration_negative] = scaler.transform(result.loc[:, migration_negative])

  # 3. Creating migration perception variables --------------------------------------------
  result["migration_perception"] = result.loc[:, migration_perception]
  result["migration_positive"] = result.loc[:, migration_positive].mean(axis=1)
  result["migration_negative"] = result.loc[:, migration_negative].mean(axis=1)
  return result

def attach_pol_pref(df: pd.DataFrame) -> pd.DataFrame:
  """
  # Creating a pol pref index

  Attaches the political preference of interviewees
  Parameters:
    df(pd.DataFrame): the wave7.csv dataframe stored by the data_filter_merge.ipynb
  
  Returns:
    pd.DataFrame: Processed dataframe with the the political preference 'pol_pref'
  """

  pol_pref = ["Q240"]

  result = df.copy()

  # 1. imputing with median --------------------------------------------
  # create a median dict with all the relevant values
  median_dict = {}
  countries = result.B_COUNTRY.unique()

  for ct in countries:
      median_dict[ct] = {}

  for tq in pol_pref:
      for ct in countries:
          median_dict[ct][tq] = result.loc[(result[tq] > 0) & (result.B_COUNTRY == ct), tq].median()
  
  # now populate the dataframe with the median values
  for tq in pol_pref:
      result[tq] = result.loc[:, [tq, "B_COUNTRY"]].apply(lambda row: median_dict[row["B_COUNTRY"]][tq] if row[tq] <= 0 else row[tq], axis=1)
    
  # 2. Minmax scaling --------------------------------------------
  scaler = MinMaxScaler()
  scaler.fit(result.loc[:, pol_pref])
  result.loc[:, pol_pref] = scaler.transform(result.loc[:, pol_pref])

  # 3. Creating political preference variable --------------------------------------------
  result["pol_pref"] = result.loc[:, pol_pref]
  return result

def attach_corruption_index(df: pd.DataFrame) -> pd.DataFrame:
  """
  # Creating a trust index

  Attaches trust indeces for baseline and group corruption to the dataframe.
  Furthermore, all trust related questions are preprocessed by imputing with the median and MinMaxScaled.

  Parameters:
    df(pd.DataFrame): the wave7.csv dataframe stored by the data_filter_merge.ipynb
  
  Returns:
    pd.DataFrame: Processed dataframe with the perception of corruption variablies in `baseline_corruption`, `group_corruption`
  """

  baseline_corruption = ["Q112"]
  corruption_questions = [f"Q{i}" for i in range(112, 120)]
  group_corruption  = [f"Q{i}" for i in range(113, 117)]

  result = df.copy()

  # 1. imputing with median --------------------------------------------
  # create a median dict with all the relevant values
  median_dict = {}
  countries = result.B_COUNTRY.unique()

  for ct in countries:
      median_dict[ct] = {}

  for tq in corruption_questions:
      for ct in countries:
          median_dict[ct][tq] = result.loc[(result[tq] > 0) & (result.B_COUNTRY == ct), tq].median()
  
  # now populate the dataframe with the median values
  for tq in corruption_questions:
      result[tq] = result.loc[:, [tq, "B_COUNTRY"]].apply(lambda row: median_dict[row["B_COUNTRY"]][tq] if row[tq] <= 0 else row[tq], axis=1)
    
  # 2. Minmax scaling --------------------------------------------
  scaler = MinMaxScaler()
  scaler.fit(result.loc[:, corruption_questions])
  result.loc[:, corruption_questions] = scaler.transform(result.loc[:, corruption_questions])

  # 3. Creating corruption variables --------------------------------------------
  result["baseline_corruption"] = result.loc[:, baseline_corruption]
  result["group_corruption"] = result.loc[:, group_corruption].mean(axis=1)


  return result

def transform_demographcis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # must have valid gender value
    df = df[df["Q260"]>0]
    df['gender'] = df['Q260'].map({1: 1, 2: 0})# 1 is male 2 is female
    # must have valid age
    df = df[df["Q262"]>0]

    df["age"] = df["Q262"]
    # must have valid values about parental and individual immigration status
    df = df[(df["Q263"]>0) & (df["Q264"]>0)&(df["Q265"]>0)]
    # create binary variables for immigration status of family and self
    df["mother_immigrant"]=df['Q263'].map({1: 0, 2: 1})
    df["father_immigrant"]=df['Q264'].map({1: 0, 2: 1})
    df["is_immigrant"]=df['Q265'].map({1: 0, 2: 1})

    return df
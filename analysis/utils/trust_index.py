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
  trust_questions = [f"Q{i}" for i in range(58, 90)]
  base_trust_qustions = [f"Q{i}" for i in range(58, 64)]
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


def attach_happiness_index(df: pd.DataFrame) -> pd.DataFrame:
    
    # Define variable groups
    happiness_questions = [f"Q{i}" for i in range(46, 56)]
    baseline_happiness = ['Q46'] 
    health = ['Q47']  
    freedom = ['Q48']  
    baseline_satisfaction = ['Q49']  
    financial_satisfaction = ['Q50']  
    hardships_questions = [f"Q{i}" for i in range(51, 56)]  # Composite index
    standard_parents = ['Q56']  # Standalone variable, categorial

    # Copy the DataFrame to work on
    result = df.copy()

    # 1. Convert 'Q56' to dummy variables -----------------------------------------
    # Replace invalid values in Q56
    result['Q56'] = result['Q56'].where(result['Q56'] > 0, pd.NA)

    # country based imputation with mode
    result['Q56'] = (
        result.groupby('B_COUNTRY_ALPHA')['Q56']
        .apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 0))
        .reset_index(level=0, drop=True)  
    )

    # Generate dummy variables for Q56 with explicit integer type
    dummies = pd.get_dummies(result['Q56'], prefix='standard_parents').astype(int)

    # Generate dummy variables for Q56 with explicit integer type
    dummies = pd.get_dummies(result['Q56'], prefix='standard_parents').astype(int)

    # Rename the dummy columns
    dummies.rename(columns={
        'standard_parents_1.0': 'standard_parents_better',
        'standard_parents_2.0': 'standard_parents_worse',
        'standard_parents_3.0': 'standard_parents_same'
    }, inplace=True)

    dummies = dummies.drop(columns=['standard_parents_same'])

    # Attach the dummy variables to the DataFrame
    result = pd.concat([result, dummies], axis=1)

    # 2. Impute missing values with the median -------------------------------------
    # Create a median dictionary for countries
    median_dict = {}
    countries = result['B_COUNTRY_ALPHA'].unique()

    # Reverse the scale for happiness questions
    for hq in baseline_happiness:
        result[hq] = result[hq].where(result[hq] <= 0, 4 + 1 - result[hq])
    # Reverse the scale for health questions
    for hq in health:
        result[hq] = result[hq].where(result[hq] <= 0, 5 + 1 - result[hq])
    # Reverse the scale for hardships questions
    for hq in hardships_questions:
        result[hq] = result[hq].where(result[hq] <= 0, 4 + 1 - result[hq])

    for ct in countries:
        median_dict[ct] = {}
        for hq in happiness_questions:
            # Calculate median for each question within each country
            median_dict[ct][hq] = result.loc[(result[hq] > 0) & (result['B_COUNTRY_ALPHA'] == ct), hq].median()

    # Populate the DataFrame with the imputed values
    for hq in happiness_questions:
        result[hq] = result.apply(
            lambda row: median_dict[row['B_COUNTRY_ALPHA']][hq] if row[hq] <= 0 else row[hq], axis=1
        )

    # 3. Normalize happiness_questions using Min-Max Scaling ---------------------
    scaler = MinMaxScaler()
    scaler.fit(result.loc[:, happiness_questions])
    result.loc[:, happiness_questions] = scaler.transform(result.loc[:, happiness_questions])

    # 4. Attach standalone and composite features ---------------------------------
    result['baseline_happiness'] = result.loc[:, baseline_happiness]
    result['health'] = result.loc[:, health]
    result['freedom'] = result.loc[:, freedom]
    result['baseline_satisfaction'] = result.loc[:, baseline_satisfaction]
    result['financial_satisfaction'] = result.loc[:, financial_satisfaction]
    # Composite variable: Hardships 
    result['hardships_questions'] = result.loc[:, hardships_questions].mean(axis=1)

    return result


def attach_security_index(df: pd.DataFrame) -> pd.DataFrame:

    # Define variable groups
    security_question = [f"Q{i}" for i in range(131, 139)] + [f"Q{i}" for i in range(142, 144)] + [f"Q{i}" for i in range(146, 149)]

    baseline_security = ['Q131']
    security_neighborhood = [f"Q{i}" for i in range(132, 139)]
    security_financial = [f"Q{i}" for i in range(142, 144)]
    security_war = [f"Q{i}" for i in range(146, 149)]
    
    # dummies
    security_actions_money = ['Q139']
    security_actions_night = ['Q140']
    security_actions_weapon = ['Q141']
    victim_respondent = ['Q144']
    victim_family = ['Q145']
    war_yes_no = ['Q151']
    yes_no = ['Q139', 'Q140', 'Q141', 'Q144', 'Q145', 'Q151']

    result = df.copy()

    # 1. Impute missing values with the median -------------------------------------
    # Create a median dictionary for countries
    median_dict = {}
    mode_dict = {}
    countries = result['B_COUNTRY'].unique()

    # Reverse the scale for baseline_security
    for tq in baseline_security:
        result[tq] = result[tq].where(result[tq] <= 0, 4 + 1 - result[tq])

    for ct in countries:
        median_dict[ct] = {}
        for tq in security_question:
            # Calculate median for each question within each country
            median_dict[ct][tq] = result.loc[(result[tq] > 0) & (result['B_COUNTRY'] == ct), tq].median()

    # Populate the DataFrame with the imputed values
    for tq in security_question:
        result[tq] = result.apply(
            lambda row: median_dict[row['B_COUNTRY']][tq] if row[tq] <= 0 else row[tq], axis=1
        )

    # 2. Normalize using Min-Max Scaling ---------------------
    scaler = MinMaxScaler()
    scaler.fit(result.loc[:, security_question])
    result.loc[:, security_question] = scaler.transform(result.loc[:, security_question])

    # 3. create DUMMIES 
    for d in yes_no:
        result[d] = result[d].where(result[d] > 0, pd.NA)

        # Country-based imputation with mode
        result[d] = (
            result.groupby('B_COUNTRY')[d]
            .apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 0))
            .reset_index(level=0, drop=True)  
        )
        # Generate dummy variables 
        dummies = pd.get_dummies(result[d], prefix=d, drop_first=True).astype(int)

        # Attach the dummy variables to the DataFrame
        result = pd.concat([result, dummies], axis=1)

    # Rename the dummy columns, this whole thing is awful help
    result.rename(columns={
        'Q139_1.0': 'security_actions_money_yes',
        'Q139_2.0': 'security_actions_money_no',
        'Q140_1.0': 'security_actions_night_yes',
        'Q140_2.0': 'security_actions_night_no',
        'Q141_1.0': 'security_actions_weapon_yes',
        'Q141_2.0': 'security_actions_weapon_no',
        'Q144_1.0': 'victim_respondent_yes',
        'Q144_2.0': 'victim_respondent_no',
        'Q145_1.0': 'victim_family_yes',
        'Q145_2.0': 'victim_family_no',
        'Q151_1.0': 'war_yes',
        'Q151_2.0': 'war_no',
    }, inplace=True)

    # 4. Attach standalone and composite features ---------------------------------
    result['baseline_security'] = result.loc[:, baseline_security]
    result['security_neighborhood'] = result.loc[:, security_neighborhood].mean(axis=1)
    result['security_financial'] = result.loc[:, security_financial].mean(axis=1)
    result['security_war'] = result.loc[:, security_war].mean(axis=1)

    return result


def attach_education_index(df: pd.DataFrame) -> pd.DataFrame:
    education = ['Q275']
    education_mother = ['Q277']
    education_father = ['Q278']

    education_questions = ['Q275', 'Q277', 'Q278']

    result = df.copy()

    # 1. Impute missing values with the median -------------------------------------
    # Create a median dictionary for countries
    median_dict = {}
    countries = result['B_COUNTRY'].unique()

    for ct in countries:
        median_dict[ct] = {}
        for e in education_questions:
            # Calculate median for each question within each country
            median_dict[ct][e] = result.loc[(result[e] >= 0) & (result['B_COUNTRY'] == ct), e].median()

    # Populate the DataFrame with the imputed values
    for e in education_questions:
        result[e] = result.apply(
            lambda row: median_dict[row['B_COUNTRY']][e] if row[e] < 0 else row[e], axis=1
        )

    # 2. Attach standalone and composite features ---------------------------------
    result['education'] = result.loc[:, education]
    result['education_mother'] = result.loc[:, education_mother]
    result['education_father'] = result.loc[:, education_father]
    
    return result

def attach_income_index(df: pd.DataFrame) -> pd.DataFrame:
    chief_wage_earner = ['Q285']
    income_group = ['Q288']

    result = df.copy()

    # 1. Impute missing values with the median -------------------------------------
    # Create a median dictionary for countries
    median_dict = {}
    countries = result['B_COUNTRY'].unique()

    for ct in countries:
        median_dict[ct] = {}
        for e in income_group:
            # Calculate median for each question within each country
            median_dict[ct][e] = result.loc[(result[e] > 0) & (result['B_COUNTRY'] == ct), e].median()

    # Populate the DataFrame with the imputed values
    for e in income_group:
        result[e] = result.apply(
            lambda row: median_dict[row['B_COUNTRY']][e] if row[e] <= 0 else row[e], axis=1
        )

    # Country-based imputation with mode
    result['Q285'] = result['Q285'].where(result['Q285'] > 0, pd.NA)

    result['Q285'] = (
        result.groupby('B_COUNTRY')['Q285']
        .apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 0))
        .reset_index(level=0, drop=True)  
    )

    # 2. Normalize using Min-Max Scaling ---------------------
    scaler = MinMaxScaler()
    scaler.fit(result.loc[:, income_group])
    result.loc[:, income_group] = scaler.transform(result.loc[:, income_group])

    # 3. Attach features ---------------------------------
    result['chief_wage_earner_yes'] = result['Q285'].map({1: 0, 2: 1}) 
    result['income_group'] = result.loc[:, income_group]

    return result
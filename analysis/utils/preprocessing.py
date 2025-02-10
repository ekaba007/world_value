import pandas as pd
import os
import openpyxl
import trust_index as ti

# Read in wave (df) and time series (ts) data
# we should keep the path general, please change dir in terminal so we dont need to change that everytime in the code
#os.chdir('/Users/carterhogan/CaseStudies/world_value_survey/analysis/data/wvs/')
# instead use in prompt 'cd C:\Users\...\world_value_survey'
print("Reading in wave data..")
df = pd.read_csv('analysis/data/wvs/WVS_Cross_National.csv')
print("Reading in time series data...")
ts = pd.read_csv('analysis/data/wvs/WVS_Time_Series.csv')

# Read in two separate instances of WPFI
print("Reading in world press freedom index...")
#os.chdir('/Users/carterhogan/CaseStudies/world_value_survey/analysis/data/additional/')
wpf = pd.read_excel('analysis/data/additional/RWB-PFI.xlsx')
wpf_ts = pd.read_excel('analysis/data/additional/RWB-PFI.xlsx')

def clean_wpfi_wave(wpf: pd.DataFrame) -> pd.DataFrame:
    countries = ['AUS','CAN','DEU','NLD','USA']
    wpf = wpf[wpf['Economy ISO3'].isin(countries)]
    # Use rank instead of index because it is the same measurement over time
    # # Index changed around 2013 to be calculated completely differently
    wpf = wpf[wpf['Indicator ID'] != 'RWB.PFI.RANK']
    wpf = pd.concat([wpf.iloc[:, :2], wpf.iloc[:, -7:]], axis=1)
    main_columns = wpf.iloc[:, :2]  # First two column
    year_columns = wpf.iloc[:, -7:]  # Last seven columns
    # pivot the last seven columns into rows so that we can merge into broader d
    pivoted_df = pd.melt(
        wpf, 
        id_vars=main_columns.columns,  # Columns to keep (e.g., the first two columns)
        value_vars=year_columns.columns,  # Columns to pivot (e.g., the last seven columns)
        var_name='Year',  # Name of the new column that will hold column names
        value_name='wpfi_rank'  # Name of the new column that will hold the values
    )
    return pivoted_df


def merge_wave_wpfi(df: pd.DataFrame, pivoted_df: pd.DataFrame) -> pd.DataFrame:
    # Filter for Countries in "Western Europe" as defined by the European Union (https://eur-lex.europa.eu/browse/eurovoc.html?params=72,7206,913#arrow_913)
    countries = ['AUS','CAN','DEU','NLD','USA']
    # Belgium, Ireland, Liechtenstein, Luxembourg, and Monaco do not have data in wave 7
    df = df[df['B_COUNTRY_ALPHA'].isin(countries)]
    # Perform the left join on 'ISO' and 'Year'
    merged_df = pd.merge(
        df, 
        pivoted_df, 
        how='left', 
        left_on=['B_COUNTRY_ALPHA', 'A_YEAR'], 
        right_on=['Economy ISO3', 'Year']
    )
    return merged_df

def clean_wpfi_ts(wpf_ts: pd.DataFrame) -> pd.DataFrame:
    main_columns = wpf_ts.iloc[:, :2]  # First two columns (Rank and Country)
    year_columns = wpf_ts.iloc[:, -21:]  # Last 21 columns (2002-2023)
    wpf_ts = wpf_ts[wpf_ts['Economy ISO3'] == "DEU"]
    wpf_ts = wpf_ts[wpf_ts['Indicator ID'] != 'RWB.PFI.RANK']
    # pivot for the time series
    pivoted_wpf = pd.melt(
        wpf_ts, 
        id_vars=main_columns.columns,  # Columns to keep (e.g., the first two columns)
        value_vars=year_columns.columns,  # Columns to pivot (e.g., the last seven columns)
        var_name='Year',  # Name of the new column that will hold column names
        value_name='wpfi_rank'  # Name of the new column that will hold the values
    )   
    return pivoted_wpf

def merge_ts_wpfi(ts: pd.DataFrame, pivoted_wpf: pd.DataFrame) -> pd.DataFrame:
    # filter just for Germany
    ts = ts[ts['COUNTRY_ALPHA']== 'DEU']
    # perform the merge
    ts_merge = pd.merge(
        ts, 
        pivoted_wpf, 
        how='left', 
        left_on=['COUNTRY_ALPHA', 'S020'], 
        right_on=['Economy ISO3', 'Year']
    )
    return ts_merge


def run_preprocessing(df: pd.DataFrame, wpf:pd.DataFrame, ts: pd.DataFrame, wpf_ts: pd.DataFrame):
    #executed data transformation
    # first pivot the wpfi data
    print("Pivoting WPFI Data...")
    pivoted_df = clean_wpfi_wave(wpf)
    pivoted_wpf = clean_wpfi_ts(wpf_ts)
    # second merge to each data set
    print("Merge WPFI Data...")
    merged_df = merge_wave_wpfi(df,pivoted_df)
    merged_ts = merge_ts_wpfi(ts,pivoted_wpf)
    print("Attaching Indices and Transforming Demographics...")
    # Attach Trust Index
    merged_df = ti.attach_distrust_index(merged_df)
    #Attach Corruption Infomation
    merged_df = ti.attach_corruption_index(merged_df)
    # Attach Migration Index
    merged_df = ti.attach_migration_index(merged_df)
    # Attach income index
    merged_df = ti.attach_income_index(merged_df)
    #Attach political preference 
    merged_df = ti.attach_pol_pref(merged_df)
    # Transform demographics
    merged_df = ti.transform_demographcis(merged_df)
    # Dummies for happiness, below and above average 
    merged_df = ti.attach_happiness_index(merged_df)
    # Attach Security Index
    merged_df = ti.attach_security_index(merged_df)
    # Attach Education Index
    merged_df = ti.attach_education_index(merged_df)
    # Attach Country dummies
    merged_df = ti.attach_corruption_index(merged_df)

    print("Writing Data to .csv files..")
    #Write the preprocessed wave data to a csv
    merged_df.to_csv('analysis/data/wvs/wave7.csv')
    #Write the preprocessed ts data to a csv
    merged_ts.to_csv('analysis/data/wvs/time_series.csv')

print("Executing Preprocessing....")
run_preprocessing(df,wpf,ts,wpf_ts)
# ======================================================
# Canopy Resistance Models - Python Implementation
# Four types of canopy resistance models
# ======================================================

# Import required libraries
import pandas as pd          # Data handling and analysis
import os                    # File and directory operations
import random                # Random number generation
from xgboost import XGBRegressor  # Machine learning model (XGBoost Regressor)
import seaborn as sns        # Data visualization
import numpy as np           # Numerical computations
from scipy import stats      # Statistical functions
import matplotlib.pyplot as plt  # Plotting library

# Set default font to Arial for all matplotlib figures
plt.rcParams['font.family'] = 'Arial'

def PenmanMonteith_rc(dlt, lambd, psy, rn, g, rho_a, vpd, et, ra):
    """
    Calculate canopy resistance (rc) based on the Penman-Monteith equation.

    Parameters
    ----------
    dlt : float
        Slope of the saturation vapor pressure curve [kPa °C-1].
    lambd : float
        Latent heat of vaporization [MJ kg-1].
    psy : float
        Psychrometric constant [kPa °C-1].
    rn : float
        Net radiation at the crop surface [MJ m-2 day-1].
    g : float
        Soil heat flux density [MJ m-2 day-1] (≈ 0 at daily scale).
    rho_a : float
        Air density [kg m-3].
    vpd : float
        Vapor pressure deficit [kPa].
    et : float
        Potential evapotranspiration [mm d-1].
    ra : float
        Aerodynamic resistance [s m-1].

    Returns
    -------
    rc : float
        Canopy resistance [s m-1].
    """

    kmin = 86400  # Seconds per day [s d-1]

    # Specific heat of moist air [MJ kg-1 °C-1]
    CP = 1.013 * 10**-3

    # Radiation term: effect of net radiation and soil heat flux
    num1 = dlt * (rn - g)

    # Aerodynamic term: effect of VPD, air density, and aerodynamic resistance
    num2 = rho_a * CP * kmin * vpd / ra

    # Rearranged Penman-Monteith formulation to solve for canopy resistance (rc)
    rc = (((num1 / et + num2 / et) / lambd - dlt) / psy - 1) * ra

    return rc
def calc_Psy(pressure, lambd):
    """
    Calculate psychrometric constant (γ).

    Parameters
    ----------
    pressure : float or array_like
        Atmospheric pressure [kPa].
    lambd : float
        Latent heat of vaporization [MJ kg-1].

    Returns
    -------
    psy : float
        Psychrometric constant [kPa °C-1].

    Notes
    -----
    - Based on FAO-56 (Allen et al., 1998), Eq. 8.
    - γ = Cp * P / (ε * λ)
        where Cp = 1.013 [kJ kg-1 °C-1], ε = 0.622.
    """
    # Cp = specific heat of moist air [kJ kg-1 °C-1]
    CP = 1.013  

    # 0.001 converts kJ to MJ
    return CP * pressure * 0.001 / (0.622 * lambd)


def calc_rho(pressure, tmean, ea):
    """
    Calculate atmospheric air density (ρa).

    Parameters
    ----------
    pressure : float
        Atmospheric pressure [kPa].
    tmean : float
        Mean daily air temperature [°C].
    ea : float
        Actual vapor pressure [kPa].

    Returns
    -------
    rho_a : float
        Air density [kg m-3].

    Notes
    -----
    - Based on FAO-56 (Allen et al., 1998), Eq. 3–5.
    - Virtual temperature (Tv) is used to correct for humidity effect:
        Tv = (T + 273.16) / (1 - 0.378 * ea / P)
    """
    # Virtual temperature [K]
    tkv = (273.16 + tmean) * (1 - 0.378 * ea / pressure) ** -1

    # Air density [kg m-3]
    return 3.486 * pressure / tkv


def calc_lambda(tmean):
    """
    Calculate latent heat of vaporization (λ).

    Parameters
    ----------
    tmean : float
        Mean daily air temperature [°C].

    Returns
    -------
    lambd : float
        Latent heat of vaporization [MJ kg-1].

    Notes
    -----
    - Approximation from FAO-56:
        λ = 2.501 - 0.002361 * T
    - At T = 20 °C, λ ≈ 2.45 MJ kg-1.
    """
    return 2.501 - 0.002361 * tmean


def calc_dlt(t):
    """
    Calculate slope of the saturation vapor pressure curve (Δ).

    Parameters
    ----------
    t : float
        Air temperature [°C].

    Returns
    -------
    dlt : float
        Slope of vapor pressure curve [kPa °C-1].

    Notes
    -----
    - Based on FAO-56 (Allen et al., 1998), Eq. 13.
    - Δ = 4098 * es(T) / (T + 237.3)^2
    """
    tmp = 4098 * (0.6108 * np.exp((17.27 * t) / (t + 237.3)))
    return tmp / ((t + 237.3) ** 2)


def calc_es(tmean):
    """
    Calculate saturation vapor pressure (es).

    Parameters
    ----------
    tmean : float
        Mean daily air temperature [°C].

    Returns
    -------
    es : float
        Saturation vapor pressure [kPa].

    Notes
    -----
    - Based on FAO-56 (Allen et al., 1998), Eq. 11.
    - es = 0.6108 * exp(17.27 * T / (T + 237.3))
    """
    return 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))
def PenmanMonteith(dlt, lambd, psy, rn, g, rho_a, vpd, rc, ra):
    """
    Calculate potential evapotranspiration (PET) using the Penman-Monteith equation.

    Parameters
    ----------
    dlt : float
        Slope of the saturation vapor pressure curve [kPa °C-1].
    lambd : float
        Latent heat of vaporization [MJ kg-1].
    psy : float
        Psychrometric constant [kPa °C-1].
    rn : float
        Net radiation at the crop surface [MJ m-2 day-1].
    g : float
        Soil heat flux density [MJ m-2 day-1] (≈ 0 at daily scale).
    rho_a : float
        Air density [kg m-3].
    vpd : float
        Vapor pressure deficit [kPa].
    rc : float
        Canopy (surface) resistance [s m-1].
    ra : float
        Aerodynamic resistance [s m-1].

    Returns
    -------
    pet : float
        Potential evapotranspiration [mm d-1].

    Notes
    -----
    - Based on FAO-56 (Allen et al., 1998).
    - Rearranged Penman-Monteith form:
        PET = [Δ(Rn - G) + (ρa * Cp * VPD / ra)] / [λ (Δ + γ (1 + rc/ra))]
    """

    kmin = 86400  # Seconds per day [s d-1]

    # Specific heat of moist air [MJ kg-1 °C-1]
    CP = 1.013 * 10**-3

    # Modified psychrometric constant considering canopy resistance
    gamma1 = psy * (1 + rc / ra)

    # Denominator: λ (Δ + γ(1 + rc/ra))
    den = lambd * (dlt + gamma1)

    # Radiation term
    num1 = dlt * (rn - g) / den

    # Aerodynamic term
    num2 = rho_a * CP * kmin * vpd / ra / den

    # Potential evapotranspiration [mm d-1]
    pet = num1 + num2

    return pet
import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_metrics(observed, simulated):
    """
    Calculate statistical performance metrics between observed and simulated values.

    Parameters
    ----------
    observed : array_like
        Observed values.
    simulated : array_like
        Simulated or predicted values.

    Returns
    -------
    r2 : float
        Coefficient of determination (R²).
    rmse : float
        Root Mean Square Error (RMSE).
    bias : float
        Mean bias (simulated - observed).
    S : float
        Modified Willmott’s index of agreement.

    Notes
    -----
    - R² is calculated as the square of Pearson correlation coefficient.
    - RMSE = sqrt(mean((obs - sim)^2)).
    - Bias = mean(sim - obs).
    - S is based on Willmott’s refined index of agreement.
    """

    # Convert input to numpy arrays
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)

    # Remove NaN values
    mask = ~np.isnan(observed) & ~np.isnan(simulated)
    observed = observed[mask]
    simulated = simulated[mask]

    # R² (coefficient of determination)
    r2 = (np.corrcoef(observed, simulated)[0, 1])**2

    # RMSE
    rmse = np.sqrt(mean_squared_error(observed, simulated))

    # Bias
    bias = np.mean(simulated - observed)

    # Modified Willmott’s index (S)
    R0 = 0.999
    correlation = np.corrcoef(observed, simulated)[0, 1]
    S = 4 * (1 + correlation)**2 / (
        (np.std(observed) / np.std(simulated) + np.std(simulated) / np.std(observed))**2 * (1 + R0)**2
    )

    return r2, rmse, bias, S
def g1(gc, co2, GPP, VPD, f_theta):
    """
    Estimate Medlyn model parameter g1 from stomatal conductance.

    Parameters
    ----------
    gc : float
        Stomatal conductance [mol m-2 s-1].
    co2 : float
        Ambient CO2 concentration [µmol mol-1].
    GPP : float
        Gross primary production [µmol m-2 s-1].
    VPD : float
        Vapor pressure deficit [kPa].
    f_theta : float
        Soil moisture stress factor (0–1).

    Returns
    -------
    g1 : float
        Medlyn slope parameter (unitless).
    
    Notes
    -----
    Based on Medlyn et al. (2011) stomatal conductance model:
        gc = 1.6 * (1 + g1 * f_theta / sqrt(VPD)) * GPP / CO2
    """
    return (gc * co2 / (1.6 * GPP) - 1) / f_theta * VPD**0.5


def gc(g1, co2, GPP, VPD, f_theta):
    """
    Estimate stomatal conductance (gc) using Medlyn model.

    Parameters
    ----------
    g1 : float
        Medlyn slope parameter (unitless).
    co2 : float
        Ambient CO2 concentration [µmol mol-1].
    GPP : float
        Gross primary production [µmol m-2 s-1].
    VPD : float
        Vapor pressure deficit [kPa].
    f_theta : float
        Soil moisture stress factor (0–1).

    Returns
    -------
    gc : float
        Stomatal conductance [mol m-2 s-1].

    Notes
    -----
    Based on Medlyn et al. (2011) stomatal conductance model:
        gc = 1.6 * (1 + f_theta * g1 / sqrt(VPD)) * GPP / CO2
    """
    return 1.6 * (1 + f_theta * g1 / VPD**0.5) * GPP / co2
# ======================================================
# FluxNet Forcing Data Processing
#   - Read site metadata and forcing data
#   - Calculate meteorological and biophysical variables
#   - Estimate canopy resistance (rc) with Penman-Monteith
# ======================================================

import pandas as pd
import numpy as np
import os

# Base path for FluxNet data
Path = '/mnt/07dd4903-cc5d-4c3c-9fed-b27b8ac045db/world/FluxNet/'

# Required columns (forcing data)
forcing_cols = [
    'TIMESTAMP', 'LE_CORR', 'H_CORR', 'NETRAD', 'NETRAD_QC', 
    'SW_IN_F', 'SW_IN_F_QC', 'LAI', 'G_F_MDS', 'G_F_MDS_QC',
    'P_F', 'P_F_QC', 'TA_F', 'TA_F_QC', 'VPD_F', 'VPD_F_QC', 
    'PA_F', 'PA_F_QC', 'WS_F', 'WS_F_QC', 
    'SWC_F_MDS_1', 'SWC_F_MDS_1_QC',
    'CO2_F_MDS', 'CO2_F_MDS_QC', 
    'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF', 'NEE_VUT_REF_QC'
]  # Note: RH not included because many sites lack it

# Site information file
excel_path = Path + "siteinfo1.csv"
df_sites = pd.read_csv(excel_path)

# Storage list for processed data
data_list = []

# Data folder containing CSV files
data_dir = Path + "data22"  
selected_sites = df_sites['SiteID'].unique()

# Process each site
for site in selected_sites:
    csv_path = os.path.join(data_dir, f"{site}.csv")

    if os.path.exists(csv_path):
        # Load site data
        df = pd.read_csv(csv_path, usecols=forcing_cols)

        # Filter invalid values: remove NaN and unreasonable values
        df = df.dropna().loc[(df >= -100).all(axis=1)]
        df = df.dropna().loc[(df.drop(columns=['TA_F', 'G_F_MDS']) >= 0).all(axis=1)]
        # Optional: df = filter_unstressed_days(df)

        # ------------------------------------------------------
        # Calculate meteorological and biophysical variables
        # ------------------------------------------------------
        df['ra'] = 208 / df['WS_F']  # Aerodynamic resistance [s m-1]
        df['dlt'] = calc_dlt(df['TA_F'].values)  # Slope of vapor pressure curve
        es = calc_es(df['TA_F'].values)          # Saturation vapor pressure [kPa]
        df['vpd'] = df['VPD_F'].values / 10      # Convert hPa to kPa
        ea = es - df['vpd']                      # Actual vapor pressure [kPa]
        df['rho_a'] = calc_rho(df['PA_F'].values, df['TA_F'].values, ea)  # Air density
        df['lambd'] = calc_lambda(df['TA_F'].values)  # Latent heat of vaporization
        df['psy'] = calc_Psy(df['PA_F'].values, df['lambd'].values)       # Psychrometric constant
        df['rn'] = 0.0864 * df['NETRAD'].values  # Net radiation [MJ m-2 day-1]
        df['g'] = 0.0864 * df['G_F_MDS'].values  # Soil heat flux [MJ m-2 day-1]
        df['rh'] = ea / es * 100                 # Relative humidity [%]

        # Latent heat flux → evapotranspiration [mm day-1]
        df['et'] = df['LE_CORR'].values * 86400 / 1e6 / df['lambd'].values  

        # Canopy resistance [s m-1]
        df['rc'] = PenmanMonteith_rc(
            df['dlt'], df['lambd'], df['psy'], df['rn'], df['g'],
            df['rho_a'], df['vpd'], df['et'], df['ra']
        )

        # ------------------------------------------------------
        # Add site metadata
        # ------------------------------------------------------
        site_info = df_sites[df_sites['SiteID'] == site].iloc[0]
        df["SiteID"] = site
        df["IGBP"] = site_info['IGBP']
        df["AI"] = site_info['AI']
        df["latitude"] = site_info['SiteLatitude']
        df["FCA"] = site_info['FCA']
        df["PWP"] = site_info['PWP']

        # Append to list
        data_list.append(df)

    else:
        print(f"File {csv_path} not found, skipping site {site}.")

# Merge all site data
if data_list:
    final_df = pd.concat(data_list, ignore_index=True)
    # print(final_df.head())  # Preview first rows
else:
    print("No valid data extracted.")
# ======================================================
# Map IGBP vegetation types to numeric codes (1-12)
# ======================================================

# Get unique IGBP categories (up to 12)
unique_classes = final_df['IGBP'].unique()[:12]

# Create mapping dictionary: category -> code (1-12)
mapping_dict = {category: i + 1 for i, category in enumerate(unique_classes)}

# Apply mapping to generate a new column 'IGBP_code'
final_df['IGBP_code'] = final_df['IGBP'].map(mapping_dict)

# Optional: check mapping
# print(final_df[['IGBP', 'IGBP_code']].drop_duplicates())
# ======================================================
# Calculate stomatal conductance, soil moisture factor, and Medlyn g1
# Filter data within reasonable ranges
# ======================================================

# Molar volume of air (Vm) [m3 mol-1]
final_df['Vm'] = 8.314 * (final_df['TA_F'] + 273.15) / final_df['PA_F'] / 1000

# Canopy stomatal conductance [mol m-2 s-1]
final_df['gc'] = 1 / final_df['rc'] / final_df['Vm']

# Soil moisture stress factor f_theta (0-1)
final_df['f_theta'] = ((final_df['SWC_F_MDS_1'] - final_df['PWP']) /
                       (final_df['FCA'] - final_df['PWP'])).clip(lower=0, upper=1)

# Medlyn stomatal slope parameter g1
final_df['g1'] = g1(
    final_df['gc'],
    final_df['CO2_F_MDS'],
    final_df['GPP_DT_VUT_REF'],
    final_df['vpd'],
    final_df['f_theta']
)

# ------------------------------------------------------
# Filter data for reasonable ranges
# - Only consider days when soil water > wilting point (f_theta > 0)
# - g1 should be positive and below 30 (typical range)
# - rc should be positive and below 5000 s m-1
# ------------------------------------------------------
final_df = final_df[final_df['f_theta'] > 0.0]
final_df = final_df[(final_df['g1'] > 0) & (final_df['g1'] < 30)]
final_df = final_df[(final_df['rc'] > 0.0) & (final_df['rc'] < 5000)]
# ======================================================
# Calculate radiation-driven evapotranspiration component (E)
# ======================================================

# E = Δ / (Δ + γ) / λ * Rn
# - Δ : slope of saturation vapor pressure curve [kPa °C-1]
# - γ : psychrometric constant [kPa °C-1]
# - λ : latent heat of vaporization [MJ kg-1]
# - Rn: net radiation [MJ m-2 day-1]

final_df['E'] = (final_df['dlt'] / (final_df['dlt'] + final_df['psy']) /
                 final_df['lambd'] * final_df['rn'])import pandas as pd

# ======================================================
# Add year and annual mean temperature per site
# ======================================================

# Ensure TIMESTAMP column is datetime type
final_df['TIMESTAMP'] = pd.to_datetime(final_df['TIMESTAMP'], format='%Y%m%d')

# Extract year from timestamp
final_df['year'] = final_df['TIMESTAMP'].dt.year

# Filter data for temperatures above 0 °C
filtered_df = final_df[final_df['TA_F'] > 0]

# Compute mean annual temperature per site
T_df = (
    filtered_df
    .groupby(['SiteID', 'year'])['TA_F']
    .mean()
    .reset_index()
    .rename(columns={'TA_F': 'T'})
)

# Merge annual mean temperature back into main dataframe
final_df = final_df.merge(T_df, on=['SiteID', 'year'], how='left')
import pandas as pd

# ======================================================
# Compute annual mean g1 per site and merge back
# ======================================================

# Ensure TIMESTAMP column is datetime type
final_df['TIMESTAMP'] = pd.to_datetime(final_df['TIMESTAMP'], format='%Y%m%d')

# Extract year from timestamp
final_df['year'] = final_df['TIMESTAMP'].dt.year

# Optional: filter out unrealistic values if needed
# filtered_df = final_df[final_df['g1'] > 0]

# Compute mean annual g1 per site
G1_df = (
    final_df
    .groupby(['SiteID', 'year'])['g1']
    .mean()
    .reset_index()
    .rename(columns={'g1': 'G1m'})
)

# Merge annual mean g1 back into main dataframe
final_df = final_df.merge(G1_df, on=['SiteID', 'year'], how='left')
import pandas as pd

# ======================================================
# Compute annual Moisture Index (MI = P / E) per site
# ======================================================

# Ensure TIMESTAMP column is datetime type
final_df['TIMESTAMP'] = pd.to_datetime(final_df['TIMESTAMP'], format='%Y%m%d')

# Extract year from timestamp
final_df['year'] = final_df['TIMESTAMP'].dt.year

# Compute annual sums of precipitation (P_F) and radiation-driven evapotranspiration (E) per site
yearly_sum = (
    final_df
    .groupby(['SiteID', 'year'])[['P_F', 'E']]
    .sum()
    .reset_index()
)

# Calculate annual Moisture Index (MI = P / E)
yearly_sum['MMI'] = yearly_sum['P_F'] / yearly_sum['E']

# Merge annual MI back into the main dataframe
final_df = final_df.merge(
    yearly_sum[['SiteID', 'year', 'MMI']],
    on=['SiteID', 'year'],
    how='left'
)
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

# ======================================================
# 5-fold Cross-validation for g1 prediction using XGBoost
# Then calculate predicted rc and ET based on predicted g1
# ======================================================

kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_dfs = []  # Store validation sets with predictions

# Features and target
features = [
    'GPP_DT_VUT_REF', 'vpd', 'TA_F', 'SW_IN_F', 'CO2_F_MDS',
    'SWC_F_MDS_1', 'LAI', 'IGBP_code'
]
X = final_df[features]
y = final_df['g1']

for train_index, test_index in kf.split(X):
    # Split train and validation sets
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    
    train_df = final_df.iloc[train_index].copy()
    
    # Optionally filter training set to avoid outliers / stressed conditions
    # train_df_filtered = train_df[
    #     (train_df['g1'] > 0) &
    #     (train_df['g1'] < 30) &
    #     (train_df['rc'] > 0) &
    #     (train_df['rc'] < 5000) &
    #     (train_df['f_theta'] > 0)
    # ].copy()
    train_df_filtered = train_df  # currently using full train set
    
    # Update X_train and y_train for modeling
    X_train = train_df_filtered[features]
    y_train = np.log(train_df_filtered['g1'])  # log-transform g1

    # -----------------------------
    # XGBoost model
    # -----------------------------
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        colsample_bytree=0.8,
        subsample=0.8,
        min_child_weight=3,
        gamma=0,
        objective='reg:squarederror',
        random_state=42,
        tree_method='hist'
    )

    # Train model
    xgb_model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = xgb_model.predict(X_val)
    
    # Store validation results
    val_df = final_df.iloc[test_index].copy()
    val_df['pre_g1'] = np.exp(y_pred)  # inverse log-transform
    test_dfs.append(val_df)

# Concatenate all folds
test_df_all = pd.concat(test_dfs).reset_index(drop=True)

# Reconstruct predicted rc from g1
test_df_all['pre_rc'] = 1 / gc(
    test_df_all['pre_g1'],
    test_df_all['CO2_F_MDS'],
    test_df_all['GPP_DT_VUT_REF'],
    test_df_all['vpd'],
    test_df_all['f_theta']
) / test_df_all['Vm']

# Compute predicted ET using Penman-Monteith
test_df_all['pre_et'] = PenmanMonteith(
    test_df_all['dlt'],
    test_df_all['lambd'],
    test_df_all['psy'],
    test_df_all['rn'],
    test_df_all['g'],
    test_df_all['rho_a'],
    test_df_all['vpd'],
    test_df_all['pre_rc'],
    test_df_all['ra']
)

# Final result
test_df_all1 = test_df_all.copy()

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import statsmodels.api as sm

# ======================================================
# 5-fold Cross-validation for annual mean g1 (G1m) using OLS
# Then compute predicted rc and PM_ET
# ======================================================

kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_dfs = []

# Features: temperature (T) and moisture index (MI)
features = ['T', 'MMI']  # 'T' = annual mean temperature, 'MMI' = annual moisture index

for train_index, test_index in kf.split(final_df):
    # Split train and validation sets
    train_df = final_df.iloc[train_index].copy()
    test_df = final_df.iloc[test_index].copy()

    # Filter training data for reasonable g1, rc, and f_theta
    train_df_filtered = train_df[
        (train_df['g1'] > 0) & 
        (train_df['g1'] < 30) & 
        (train_df['rc'] > 0.0) & 
        (train_df['rc'] < 5000) & 
        (train_df['f_theta'] > 0.0)
    ].copy()
    
    # Construct interaction term T * MI
    train_df_filtered['T_MI'] = train_df_filtered['T'] * train_df_filtered['MMI']
    test_df['T_MI'] = test_df['T'] * test_df['MMI']

    # Construct design matrix for OLS
    X_train = train_df_filtered[['T', 'MMI', 'T_MI']]
    X_train = sm.add_constant(X_train)  # Add intercept
    y_train = np.log(train_df_filtered['G1m'])  # Log-transform target

    # Remove inf or NaN values
    train_data = pd.concat([X_train, y_train], axis=1)
    train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_data.dropna(inplace=True)

    X_train = train_data.drop(columns=[y_train.name])
    y_train = train_data[y_train.name]

    # Fit OLS model
    model = sm.OLS(y_train, X_train).fit()

    # Predict on validation set
    X_val = test_df[['T', 'MMI', 'T_MI']]
    X_val = sm.add_constant(X_val)
    y_pred_log = model.predict(X_val)
    y_pred = np.exp(y_pred_log)  # inverse log-transform

    # Save validation predictions
    val_df = test_df.copy()
    val_df['pre_g1'] = y_pred
    test_dfs.append(val_df)

# Merge all folds
test_df_all = pd.concat(test_dfs).reset_index(drop=True)

# Reconstruct predicted rc from g1
test_df_all['pre_rc'] = 1 / gc(
    test_df_all['pre_g1'],
    test_df_all['CO2_F_MDS'],
    test_df_all['GPP_DT_VUT_REF'],
    test_df_all['vpd'],
    test_df_all['f_theta']
) / test_df_all['Vm']

# Drop rows with NaN predicted rc
test_df_all = test_df_all.dropna(subset=['pre_rc']).reset_index(drop=True)

# Save final OLS-based predictions
test_df_all2 = test_df_all.copy()
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# ======================================================
# Jarvis stomatal resistance model
# ======================================================
def jarvis_model(df, rsmin):
    """
    Calculate canopy stomatal resistance using Jarvis-type model.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe containing required columns: LAIe, SW_IN_F, Rdsdbl, LAI, F2-F5
    rsmin : float
        Minimum stomatal resistance
    
    Returns
    -------
    rc : ndarray
        Predicted canopy resistance
    """
    LAIe = df['LAIe']
    rsmax = 5000
    
    # Light response factor F1
    f = 0.55 * 2 * df['SW_IN_F'] / df['Rdsdbl'] / df['LAI']
    F1 = np.clip(np.nan_to_num((f + rsmin / rsmax) / (f + 1), nan=1.0), 0, 1)

    # Other stress factors clipped to [0,1]
    F2 = df['F2'].clip(0,1)
    F3 = df['F3'].clip(0,1)
    F4 = df['F4'].clip(0,1)
    F5 = df['F5'].clip(0,1)
    
    rc = rsmin / (LAIe * F1 * F2 * F3 * F4 * F5 + 1e-6)
    return rc

# ======================================================
# Preprocessing and initialize Jarvis model factors
# ======================================================
final_df['LAIe'] = final_df['LAI']
final_df.loc[(final_df['LAI'] > 2) & (final_df['LAI'] < 4), 'LAIe'] = 2
final_df.loc[final_df['LAI'] >= 4, 'LAIe'] = 0.5 * final_df['LAI']

forest_types = ['DBF', 'EBF', 'ENF', 'MF']
final_df['Rdsdbl'] = np.where(final_df['IGBP'].isin(forest_types), 30, 100)

# Stress factors
final_df['F2'] = np.exp(-0.5 * final_df['vpd'])
final_df['F3'] = (1 - 0.0016 * (24.84 - final_df['TA_F'])**2).clip(0)
final_df['F4'] = ((final_df['SWC_F_MDS_1'] - final_df['PWP']) / 
                  (final_df['FCA'] - final_df['PWP']))
final_df['F4'] = np.where(final_df['SWC_F_MDS_1'] >= final_df['FCA'], 1, final_df['F4'])
final_df['F4'] = np.where(final_df['SWC_F_MDS_1'] <= final_df['PWP'], 0, final_df['F4'])

x = 0.9
final_df['F5'] = 1 - (1 - x)/900 * final_df['CO2_F_MDS']
final_df['F5'] = np.where(final_df['CO2_F_MDS'] <= 100, 1, final_df['F5'])
final_df['F5'] = np.where(final_df['CO2_F_MDS'] >= 1000, x, final_df['F5'])

# ======================================================
# Monte Carlo + K-Fold Cross-validation for rcmin
# ======================================================
rcmin_candidates = np.random.uniform(1, 200, size=10000)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(kf.split(final_df))
test_dfs = []

for fold_id, (train_idx, test_idx) in enumerate(folds):
    df_train = final_df.iloc[train_idx].reset_index(drop=True)
    df_test  = final_df.iloc[test_idx].reset_index(drop=True)

    for igbp in df_train['IGBP'].unique():
        print(f"Fold {fold_id}, IGBP: {igbp}")
        df_igbp = df_train[df_train['IGBP'] == igbp]
        y_train = df_igbp['rc']

        # Monte Carlo: try different rcmin candidates
        errors = [(rc, mean_squared_error(y_train, jarvis_model(df_igbp, rc))) 
                  for rc in rcmin_candidates]
        top10_rc = [rc for rc, _ in sorted(errors, key=lambda x: x[1])[:10]]
        best_rcmin = np.mean(top10_rc)

        # Predict on test set for this IGBP
        mask = df_test['IGBP'] == igbp
        df_test.loc[mask, 'pre_rc'] = jarvis_model(df_test.loc[mask], best_rcmin)
        df_test.loc[mask, 'pre_et'] = PenmanMonteith(
            df_test.loc[mask, 'dlt'],
            df_test.loc[mask, 'lambd'],
            df_test.loc[mask, 'psy'],
            df_test.loc[mask, 'rn'],
            df_test.loc[mask, 'g'],
            df_test.loc[mask, 'rho_a'],
            df_test.loc[mask, 'vpd'],
            df_test.loc[mask, 'pre_rc'],
            df_test.loc[mask, 'ra']
        )

    test_dfs.append(df_test)

# Concatenate all folds
test_df_all = pd.concat(test_dfs).reset_index(drop=True)
test_df_all3 = test_df_all.copy()
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

# ======================================================
# 5-fold Cross-validation for log(rc) prediction using XGBoost
# Then compute predicted rc and ET
# ======================================================

# Log-transform rc for stability
final_df['log_rc'] = np.log(final_df['rc'])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_dfs = []

# Predictor features
features = [
    'GPP_DT_VUT_REF', 'vpd', 'TA_F', 'SW_IN_F', 
    'CO2_F_MDS', 'SWC_F_MDS_1', 'LAI', 'IGBP_code'
]

X = final_df[features]
y = final_df['log_rc']

for train_index, test_index in kf.split(X):
    # Split train and validation sets
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    
    train_df = final_df.iloc[train_index]
    
    # Filter training data to avoid stressed or invalid conditions
    train_df_filtered = train_df[
        (train_df['g1'] > 0) & 
        (train_df['g1'] < 30) & 
        (train_df['rc'] > 0.0) & 
        (train_df['rc'] < 5000) & 
        (train_df['f_theta'] > 0.0)
    ]
    
    # Update training features and target
    X_train = train_df_filtered[features]
    y_train = train_df_filtered['log_rc']
    
    # -----------------------------
    # XGBoost regression
    # -----------------------------
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        colsample_bytree=0.8,
        subsample=0.8,
        min_child_weight=3,
        gamma=0,
        objective='reg:squarederror',
        random_state=42,
        tree_method='hist'
    )

    # Train model
    xgb_model.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred = xgb_model.predict(X_val)
    
    # Store validation results
    val_df = final_df.iloc[test_index].copy()
    val_df['pre_logrc'] = y_pred
    test_dfs.append(val_df)

# Merge all validation folds
test_df_all = pd.concat(test_dfs).reset_index(drop=True)

# Transform back to rc
test_df_all['pre_rc'] = np.exp(test_df_all['pre_logrc'])

# Compute predicted ET using Penman-Monteith
test_df_all['pre_et'] = PenmanMonteith(
    test_df_all['dlt'],
    test_df_all['lambd'],
    test_df_all['psy'],
    test_df_all['rn'],
    test_df_all['g'],
    test_df_all['rho_a'],
    test_df_all['vpd'],
    test_df_all['pre_rc'],
    test_df_all['ra']
)

# Final XGBoost-based predicted dataset
test_df_all4 = test_df_all.copy()
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, linregress
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------
# Figure layout
# ---------------------------
n_rows, n_cols = 4, 6
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12), constrained_layout=True, dpi=600)

# Models and vegetation classes
rc_classes = [r'r$_{\mathbf{c-Jarvis}}$', r'r$_{\mathbf{c-USO-Lin}}$', r'r$_{\mathbf{c-ML}}$', r'r$_{\mathbf{c-USO-ML}}$']
igbp_classes = ['ENF', 'GRA', 'CRO', 'SAV', 'DBF', 'OSH']

# Order labels for subplots
order = ['(a-1)','(a-2)','(a-3)','(a-4)','(a-5)','(a-6)',
         '(b-1)','(b-2)','(b-3)','(b-4)','(b-5)','(b-6)',
         '(c-1)','(c-2)','(c-3)','(c-4)','(c-5)','(c-6)',
         '(d-1)','(d-2)','(d-3)','(d-4)','(d-5)','(d-6)']

# Store metrics for GPI
gpi_data = {igbp: [] for igbp in igbp_classes}

# Loop over each model (rows) and vegetation type (columns)
for i_row, test_df_all in enumerate(test_df_all_list):
    for i_col, igbp in enumerate(igbp_classes):
        idx = i_row * n_cols + i_col
        ax = axes.flat[idx]

        subset = test_df_all[test_df_all['IGBP'] == igbp]
        y_true = subset['rc'].values
        y_pred = subset['pre_rc'].values

        if len(y_true) == 0:  # Skip empty subsets
            ax.axis('off')
            continue

        # Scatter density for visualization
        xy = np.vstack([y_true, y_pred])
        z = gaussian_kde(xy)(xy)
        idx_sort = z.argsort()
        x, y, z = y_true[idx_sort], y_pred[idx_sort], z[idx_sort]

        ax.scatter(x, y, s=20, color='blue', edgecolor='none')
        ticks1 = np.linspace(0, 5000, 6)
        ax.plot(ticks1, ticks1, '--', color='black', lw=2.5)  # 1:1 line
        slope, intercept, _, _, _ = linregress(x, y)
        ax.plot(ticks1, intercept + slope * ticks1, '--', color='red', lw=3)  # regression line

        # Add subplot label
        ax.text(0.01, 1.15, order[idx], transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left')

        # Compute metrics
        r2, rmse, bias, nse = calculate_metrics(y_true, y_pred)
        gpi_data[igbp].append({'Model': rc_classes[i_row], 'R2': r2, 'abs MB': abs(bias),
                               'Mean Bias': bias, 'RMSE': rmse, 'TSS': nse})

        # Axis formatting
        ax.set_xlim(-5, 5000)
        ax.set_ylim(-5, 5000)
        ax.tick_params(labelsize=20)

        if i_row < n_rows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xticks(ticks1)
            ax.set_xticklabels([f'{int(tick)}' for tick in ticks1], rotation=30)

        if i_col != 0:
            ax.set_yticklabels([])

        # Column/row titles
        if i_row == 0:
            ax.text(0.5, 1.15, igbp, transform=ax.transAxes, fontsize=24, ha='center', fontweight='bold', va='bottom')
        if i_col == 0:
            ax.text(-0.45, 0.5, rc_classes[i_row], transform=ax.transAxes, fontsize=24, ha='left',
                    va='center', fontweight='bold', rotation=90)

# Compute GPI for each vegetation type
for igbp in igbp_classes:
    metrics = pd.DataFrame(gpi_data[igbp])
    norm_metrics = metrics.copy()
    for col in ['R2', 'abs MB', 'RMSE', 'TSS']:
        min_val, max_val = metrics[col].min(), metrics[col].max()
        norm_metrics[col + '_norm'] = 0 if max_val - min_val == 0 else (metrics[col] - min_val) / (max_val - min_val)
    norm_metrics['GPI'] = (norm_metrics['R2_norm'] - np.median(norm_metrics['R2_norm']) +
                           np.median(norm_metrics['RMSE_norm']) - norm_metrics['RMSE_norm'] +
                           np.median(norm_metrics['abs MB_norm']) - norm_metrics['abs MB_norm'] +
                           norm_metrics['TSS_norm'] - np.median(norm_metrics['TSS_norm']))
    for i, row in norm_metrics.iterrows():
        gpi_data[igbp][i]['GPI'] = row['GPI']

# Add GPI text to each subplot
for i_row, test_df_all in enumerate(test_df_all_list):
    for i_col, igbp in enumerate(igbp_classes):
        idx = i_row * n_cols + i_col
        ax = axes.flat[idx]
        current_model = rc_classes[i_row]
        current_gpi = next((d['GPI'] for d in gpi_data[igbp] if d['Model'] == current_model), None)
        if current_gpi is not None:
            ax.text(1, 1.03, f'GPI = {current_gpi:.2f}', transform=ax.transAxes,
                    fontsize=20, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0., edgecolor='none'), color='red')

# Add global axis labels
fig.text(0.5, -0.02, 'Observed r$_{c}$ (s/m)', fontsize=24, ha='center', va='center')
fig.text(-0.013, 0.5, 'Simulated r$_{c}$ (s/m)', fontsize=24, ha='center', va='center', rotation='vertical')

plt.show()
import pandas as pd

# -------------------------------------------------------------
# Convert the gpi_data dictionary into a flat DataFrame
# Each row corresponds to one IGBP type and one model's metrics
# -------------------------------------------------------------
records = []

for igbp, model_list in gpi_data.items():
    for model_metrics in model_list:
        record = {
            'IGBP': igbp,                       # Vegetation type
            'Model': model_metrics['Model'],    # Model name
            'R2': model_metrics['R2'],          # Coefficient of determination
            'RMSE': model_metrics['RMSE'],      # Root Mean Squared Error
            'Bias': model_metrics['Mean Bias'], # Mean bias
            'TSS': model_metrics['TSS'],        # Total Skill Score or Nash-Sutcliffe
            'GPI': model_metrics['GPI']         # General Performance Index
        }
        records.append(record)

# Create a DataFrame from the list of records
gpi_df = pd.DataFrame(records)

# Optional: sort by IGBP type and model name
gpi_df = gpi_df.sort_values(by=['IGBP', 'Model']).reset_index(drop=True)

# Display the first few rows
print(gpi_df.head())

# Optional: save to CSV for further analysis
# gpi_df.to_csv('GPI_metrics_by_IGBP_and_Model.csv', index=False)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -------------------------
# Plot settings
# -------------------------
plt.style.use('default')
plt.rcParams.update({'font.size': 12})

# Model order and custom colors
rc_classes = [
    r'r$_{\mathbf{c-Jarvis}}$',
    r'r$_{\mathbf{c-USO-Lin}}$',
    r'r$_{\mathbf{c-ML}}$',
    r'r$_{\mathbf{c-USO-ML}}$'
]
custom_palette = {
    rc_classes[0]: '#FFE4C4',
    rc_classes[1]: '#CAFF70',
    rc_classes[2]: '#76EEC6',
    rc_classes[3]: '#FF4040'
}

# Metrics and their labels
metric_cols = ['R2', 'RMSE', 'Bias', 'TSS', 'GPI']
metric_labels = [r'$R^2$', 'RMSE (s/m)', 'MB (s/m)', 'TSS', 'GPI']
igbp_order = ['ENF', 'GRA', 'CRO', 'SAV', 'DBF', 'OSH']

# Y-axis ticks and limits for each metric
yticks_list = [
    [0, 0.2, 0.4, 0.6, 0.8, 1],
    [0, 200, 400, 600, 800, 1000],
    [-200, -100, 0, 100, 200],
    [0, 0.2, 0.4, 0.6, 0.8, 1],
    [-0.5, -0.25, 0, 0.25, 0.5]
]
ylims_list = [
    (0, 1),
    (0, 1000),
    (-200, 200),
    (0, 1),
    (-0.5, 0.5)
]
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']

# -------------------------
# Create figure and axes
# -------------------------
fig, axes = plt.subplots(len(metric_cols), 1, figsize=(10, 8), dpi=600, sharey=False)

# -------------------------
# Plot each metric as a bar chart
# -------------------------
bars_for_legend = []  # Store legend handles
for i, metric in enumerate(metric_cols):
    ax = axes[i]
    
    # Draw barplot
    sns_barplot_obj = sns.barplot(
        data=gpi_df, x='IGBP', y=metric, hue='Model',
        hue_order=rc_classes, order=igbp_order,
        ax=ax, palette=custom_palette
    )
    
    # Save legend handles and labels from first subplot
    if i == 0:
        bars_for_legend, legend_labels = ax.get_legend_handles_labels()
    
    # Set y-axis ticks and limits
    ax.set_yticks(yticks_list[i])
    ax.set_ylim(ylims_list[i])
    
    # Set axis labels
    ax.set_ylabel(metric_labels[i], fontsize=12)
    ax.set_xlabel('')
    if i < len(metric_cols) - 1:
        ax.set_xticklabels([])  # Hide x-axis labels except for bottom subplot
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Add subplot label in the top-left corner
    ax.text(0.94, 0.98, subplot_labels[i], transform=ax.transAxes,
            fontsize=12, fontweight='bold', ha='left', va='top')
    
    # Remove individual subplot legend
    ax.get_legend().remove()

# -------------------------
# Add a single legend on top
# -------------------------
fig.legend(
    handles=bars_for_legend,
    labels=legend_labels,
    loc='upper center',
    ncol=len(rc_classes),
    fontsize=11,
    frameon=False,
    bbox_to_anchor=(0.55, 0.98)
)

# Adjust layout to leave space for the top legend
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

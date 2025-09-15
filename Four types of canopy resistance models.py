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

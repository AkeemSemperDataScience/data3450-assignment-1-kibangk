
import pandas as pd
import numpy as np
import math

def age_splitter(df, col_name, age_threshold):
    """
    Splits the dataframe into two dataframes based on an age threshold.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    col_name (str): The name of the column containing age values.
    age_threshold (int): The age threshold for splitting.

    Returns:
    tuple: A tuple containing two dataframes:
        - df_below: DataFrame with rows where age is below the threshold.
        - df_above_equal: DataFrame with rows where age is above or equal to the threshold.
    """
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("LabourTrainingEvaluationData.csv")
def age_splitter(df: pd.DataFrame, col: str, threshold: int):
    below = df[df[col] < threshold].copy()
    above_equal = df[df[col] >= threshold].copy()
    return below, above_equal
below_30, above_equal_30 = age_splitter(df, "Age", 30)
mean_below = below_30["Earnings_1978"].mean()
mean_above = above_equal_30["Earnings_1978"].mean()

print(f"Mean Earnings (1978) below 30: {mean_below:.2f}")
print(f"Mean Earnings (1978) 30 and above: {mean_above:.2f}")

if mean_below > mean_above:
    print("→ People below 30 earned more on average in 1978.")
else:
    print("→ People 30 and above earned more on average in 1978.")

# ---- Visual comparison ----
plt.bar(["Below 30", "30 and above"], [mean_below, mean_above], color=["skyblue", "orange"])
plt.ylabel("Mean Earnings (1978)")
plt.title("Comparison of 1978 Earnings by Age Group")
plt.show()
    
    
def effectSizer(df, num_col, cat_col):
    """
    Calculates the effect sizes of binary categorical classes on a numerical value.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    num_col (str): The name of the numerical column.
    cat_col (str): The name of the binary categorical column.

    Returns:
    float: Cohen's d effect size between the two groups defined by the categorical column.
    Raises:
    ValueError: If the categorical column does not have exactly two unique values.
    """
    passdef effect_size_by_category(df, num_col, cat_col):
    results = {}
    categories = df[cat_col].dropna().unique()
    
    for cat in categories:
        binary_col = to_binary(df[cat_col], cat)
        g1 = df[binary_col == 1][num_col].dropna()
        g0 = df[binary_col == 0][num_col].dropna()
        
        if len(g1) > 0 and len(g0) > 0:
            d = compute_cohens_d(g1, g0)
            results[cat] = d
    
    if results:
        # Find category with largest absolute effect size
        largest_cat = max(results, key=lambda k: abs(results[k]))
        return largest_cat, results[largest_cat]
    else:
        return None, np.nan

def cohenEffectSize(group1, group2):
    # You need to implement this helper function
    # This should not be too hard...
    pass

def cohortCompare(df, cohorts, statistics=['mean', 'median', 'std', 'min', 'max']):
    """
    This function takes a dataframe and a list of cohort column names, and returns a dictionary
    where each key is a cohort name and each value is an object containing the specified statistics
    """
    pass
  

class CohortMetric():
    # don't change this
    def __init__(self, cohort_name):
        self.cohort_name = cohort_name
        self.statistics = {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None
        }
    def setMean(self, new_mean):
        self.statistics["mean"] = new_mean
    def setMedian(self, new_median):
        self.statistics["median"] = new_median
    def setStd(self, new_std):
        self.statistics["std"] = new_std
    def setMin(self, new_min):
        self.statistics["min"] = new_min
    def setMax(self, new_max):
        self.statistics["max"] = new_max

    def compare_to(self, other):
        for stat in self.statistics:
            if not self.statistics[stat].equals(other.statistics[stat]):
                return False
        return True
    def __str__(self):
        output_string = f"\nCohort: {self.cohort_name}\n"
        for stat, value in self.statistics.items():
            output_string += f"\t{stat}:\n{value}\n"
            output_string += "\n"
        return output_string

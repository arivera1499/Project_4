import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have the `coeff_df` DataFrame from earlier which contains the feature names and coefficients
# If not, create the DataFrame using the features and their respective coefficients:

# Example DataFrame creation (replace this with your own DataFrame if already available):
data = {
    'Feature': ['Housing_Index', 'Price_x', 'Rate', 'BBK_Index', '10 Yr', '1 Yr', 'INDPRO', '2 Yr', '6 Mo', '20 Yr',
                '30 Yr', '3 Yr', 'CPI', '3 Mo', 'GDP', 'Unnamed: 0', '7 Yr', '5 Yr', '4 Mo'],
    'Coefficient': [1.741345, -1.595144, 1.339536, 1.028806, 0.480520, -0.477382, -0.446273, -0.396573, -0.362975,
                    -0.357191, 0.316897, -0.315156, 0.257301, -0.225649, -0.201534, -0.149283, 0.139701, -0.113489,
                    -0.107449]
}

coeff_df = pd.DataFrame(data)

# Sorting the DataFrame by the absolute value of coefficients for better visualization
coeff_df['abs_coefficient'] = coeff_df['Coefficient'].abs()
coeff_df = coeff_df.sort_values(by='abs_coefficient', ascending=False)

# Drop the 'abs_coefficient' after sorting (optional)
coeff_df = coeff_df.drop('abs_coefficient', axis=1)

# 1. Create a horizontal bar plot
plt.figure(figsize=(10, 8))
plt.barh(coeff_df['Feature'], coeff_df['Coefficient'], color=['green' if x > 0 else 'red' for x in coeff_df['Coefficient']])
plt.xlabel('Coefficient Value')
plt.title('Feature Importance Based on Logistic Regression Coefficients')
plt.gca().invert_yaxis()  # Highest coefficients at the top

# 2. Display the plot
plt.show()

import pandas as pd
import numpy as np

# ----------------------------
# Step 0: Create Play Tennis dataset
# ----------------------------

data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny',
                'Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal','High',
                 'Normal','Normal','Normal','High','Normal','High'],
    'Play': ['No','No','Yes','Yes','Yes','No','Yes','No',
             'Yes','Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)

# ----------------------------
# Step 1: Calculate Entropy of target (Play)
# ----------------------------

yes = np.sum(df['Play'] == 'Yes')
no = np.sum(df['Play'] == 'No')

p_yes = yes / len(df)
p_no = no / len(df)

# Entropy formula
entropy_total = -p_yes*np.log2(p_yes) - p_no*np.log2(p_no)

print("Total Entropy:", entropy_total)


# ----------------------------
# Step 2: Calculate Information Gain of Outlook
# ----------------------------

values = df['Outlook'].unique()
weighted_entropy = 0

for val in values:
    
    # Take subset where Outlook = val
    subset = df[df['Outlook'] == val]
    
    yes = np.sum(subset['Play'] == 'Yes')
    no = np.sum(subset['Play'] == 'No')
    
    total = len(subset)
    
    # If pure node → entropy = 0
    if yes == 0 or no == 0:
        entropy = 0
    else:
        p_yes = yes / total
        p_no = no / total
        entropy = -p_yes*np.log2(p_yes) - p_no*np.log2(p_no)
    
    # Weighted entropy
    weighted_entropy += (total/len(df)) * entropy

# Information Gain
info_gain = entropy_total - weighted_entropy

print("Information Gain (Outlook):", info_gain)


# ----------------------------
# Step 1: Calculate Gini for Outlook
# ----------------------------

values = df['Outlook'].unique()
weighted_gini = 0

for val in values:
    
    # Take subset
    subset = df[df['Outlook'] == val]
    
    yes = np.sum(subset['Play'] == 'Yes')
    no = np.sum(subset['Play'] == 'No')
    
    total = len(subset)
    
    p_yes = yes / total
    p_no = no / total
    
    # Gini formula
    gini = 1 - (p_yes**2 + p_no**2)
    
    # Weighted gini
    weighted_gini += (total/len(df)) * gini

print("Gini Index (Outlook):", weighted_gini)
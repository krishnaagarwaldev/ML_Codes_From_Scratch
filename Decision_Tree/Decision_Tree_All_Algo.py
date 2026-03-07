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
# Step 1: Calculate Total Entropy
# ----------------------------

yes = np.sum(df['Play'] == 'Yes')
no = np.sum(df['Play'] == 'No')

p_yes = yes / len(df)
p_no = no / len(df)

entropy_total = -p_yes*np.log2(p_yes) - p_no*np.log2(p_no)

print("Total Entropy:", entropy_total)


# ============================================================
# ID3 → Information Gain for ALL features
# ============================================================

features = ['Outlook', 'Humidity']

print("\n----- INFORMATION GAIN -----")

for feature in features:
    
    values = df[feature].unique()
    print("\nFeature:", feature)
    print("Values:", values)

    weighted_entropy = 0
    
    for val in values:
        
        subset = df[df[feature] == val]
        print("Subset for", feature, "=", val)
        print(subset)

        yes = np.sum(subset['Play'] == 'Yes')
        no = np.sum(subset['Play'] == 'No')
        
        total = len(subset)
        
        if yes == 0 or no == 0:
            entropy = 0
        else:
            p_yes = yes / total
            p_no = no / total
            entropy = -p_yes*np.log2(p_yes) - p_no*np.log2(p_no)
            
        weighted_entropy += (total/len(df)) * entropy
    
    info_gain = entropy_total - weighted_entropy
    
    print(feature, "Information Gain:", info_gain)


# ============================================================
# C4.5 → Gain Ratio for ALL features
# ============================================================

print("\n----- GAIN RATIO -----")

for feature in features:
    
    values = df[feature].unique()
    weighted_entropy = 0
    split_info = 0
    
    for val in values:
        
        subset = df[df[feature] == val]
        total = len(subset)
        
        yes = np.sum(subset['Play'] == 'Yes')
        no = np.sum(subset['Play'] == 'No')
        
        # Entropy
        if yes == 0 or no == 0:
            entropy = 0
        else:
            p_yes = yes / total
            p_no = no / total
            entropy = -p_yes*np.log2(p_yes) - p_no*np.log2(p_no)
        
        weighted_entropy += (total/len(df)) * entropy
        
        # Split Info
        p = total / len(df)
        split_info -= p * np.log2(p)
    
    info_gain = entropy_total - weighted_entropy
    
    if split_info == 0:
        gain_ratio = 0
    else:
        gain_ratio = info_gain / split_info
    
    print(feature, "Gain Ratio:", gain_ratio)


# ============================================================
# CART → Gini Index for ALL features
# ============================================================

print("\n----- GINI INDEX -----")

for feature in features:
    
    values = df[feature].unique()
    weighted_gini = 0
    
    for val in values:
        
        subset = df[df[feature] == val]
        
        yes = np.sum(subset['Play'] == 'Yes')
        no = np.sum(subset['Play'] == 'No')
        
        total = len(subset)
        
        p_yes = yes / total
        p_no = no / total
        
        gini = 1 - (p_yes**2 + p_no**2)
        
        weighted_gini += (total/len(df)) * gini
    
    print(feature, "Gini Index:", weighted_gini)
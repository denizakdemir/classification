import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full dataset and drop duplicates
full = pd.read_csv('datasets/heart-2.csv').drop_duplicates()

# Shuffle and split
temp, test = train_test_split(full, test_size=0.15, random_state=42, shuffle=True)
train, val = train_test_split(temp, test_size=0.1765, random_state=42, shuffle=True)  # 0.1765*0.85 ≈ 0.15

# Save splits
train.to_csv('datasets/heart_train.csv', index=False)
val.to_csv('datasets/heart_val.csv', index=False)
test.to_csv('datasets/heart_test.csv', index=False)

print(f"Train size: {len(train)}")
print(f"Val size: {len(val)}")
print(f"Test size: {len(test)}") 
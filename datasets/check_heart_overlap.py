import pandas as pd

train = pd.read_csv('datasets/heart_train.csv')
val = pd.read_csv('datasets/heart_val.csv')
test = pd.read_csv('datasets/heart_test.csv')

# Remove index for comparison
train_nidx = train.reset_index(drop=True)
val_nidx = val.reset_index(drop=True)
test_nidx = test.reset_index(drop=True)

# Check overlap between train and val
overlap_train_val = pd.merge(train_nidx, val_nidx, how='inner')
print(f"Overlap between train and val: {len(overlap_train_val)} rows")
if len(overlap_train_val) > 0:
    print(overlap_train_val.head())

# Check overlap between train and test
overlap_train_test = pd.merge(train_nidx, test_nidx, how='inner')
print(f"Overlap between train and test: {len(overlap_train_test)} rows")
if len(overlap_train_test) > 0:
    print(overlap_train_test.head())

# Check overlap between val and test
overlap_val_test = pd.merge(val_nidx, test_nidx, how='inner')
print(f"Overlap between val and test: {len(overlap_val_test)} rows")
if len(overlap_val_test) > 0:
    print(overlap_val_test.head()) 
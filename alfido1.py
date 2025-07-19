import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# 1. Load your dataset
df = pd.read_csv(r"D:\my python stuff\data.csv") #path of folder where .csv file is stored

# 2. Handle missing values
# Numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
num_imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

# Categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# 3. Separate target variable BEFORE encoding
y = df['Purchased']  # Target column
X = df.drop(columns=['Purchased'], errors='ignore')  # won't raise error if already dropped

# 4. Encode categorical features
# Encode target variable if it's categorical
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# One-hot encode the remaining categorical features
df = pd.get_dummies(df, columns=categorical_cols.drop('Purchased', errors='ignore'))

# 5. Feature Scaling
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# Debugging print (optional)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

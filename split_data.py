# Split Customer Churn.csv into train and test sets (human-style)
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Load the full dataset
    df = pd.read_csv('Customer Churn.csv')
    print(f"Loaded data with shape: {df.shape}")

    # Stratified split for balanced churn classes
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['Churn']
    )
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Save to CSV
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    print("train.csv and test.csv created!")

if __name__ == "__main__":
    main()

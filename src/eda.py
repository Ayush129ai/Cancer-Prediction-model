import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_data

def perform_eda(X, y):
    """Perform exploratory data analysis."""
    df = X.copy()
    df['target'] = y

    # Summary statistics
    print(df.describe())

    # Class distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x='target', data=df)
    plt.title('Class Distribution')
    plt.savefig('notebooks/class_distribution.png')
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12,10))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('notebooks/correlation_heatmap.png')
    plt.show()

    # Histograms of first few features
    df.drop('target', axis=1).hist(bins=20, figsize=(20,15))
    plt.savefig('notebooks/feature_histograms.png')
    plt.show()

if __name__ == "__main__":
    X, y = load_data()
    perform_eda(X, y)
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_filter_data():
    df = pd.read_csv('youtube.csv')
    df = df[['title', 'category']].dropna()

    X = df['title'].values
    y = pd.get_dummies(df['category']).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return df, X_train, X_test, y_train, y_test

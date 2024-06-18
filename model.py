import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def load_data(file_path):
    """Load dataset from file."""
    return pd.read_csv("C:\\Users\\harih\\OneDrive\\Desktop\\CODSOFT\\TASSK-4\\Churn_Modelling.csv")

def preprocess_data(data):
    """Preprocess the data."""
    data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    X = data.drop('Exited', axis=1)
    y = data['Exited']
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])
    
    X_processed = preprocessor.fit_transform(X)
    
    # Save preprocessor
    joblib.dump(preprocessor, 'preprocessor.pkl')
    
    return X_processed, y

def train_model(X, y, model, param_grid, cv=3):
    """Train model using GridSearchCV."""
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    return best_params, best_score

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test data."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return accuracy, conf_matrix, class_report

def save_model(model, file_path):
    """Save trained model to file."""
    joblib.dump(model, file_path)
    print("Model saved successfully.")

def main():
    # Load data
    file_path = "Churn_Modelling.csv"
    data = load_data(file_path)
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models and hyperparameters for tuning
    log_reg = LogisticRegression(random_state=42)
    rand_forest = RandomForestClassifier(random_state=42)
    grad_boost = GradientBoostingClassifier(random_state=42)
    gbc_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.05],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Train models and select best parameters
    best_params, best_score = train_model(X_train, y_train, GradientBoostingClassifier(), gbc_param_grid)
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation accuracy: {best_score}")
    
    # Train final model with best parameters
    final_model = GradientBoostingClassifier(**best_params, random_state=42)
    final_model.fit(X_train, y_train)
    
    # Evaluate final model
    accuracy, conf_matrix, class_report = evaluate_model(final_model, X_test, y_test)
    print("Final Model Evaluation")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    save_model(final_model, 'customer_churn_model.pkl')

if __name__ == "__main__":
    main()

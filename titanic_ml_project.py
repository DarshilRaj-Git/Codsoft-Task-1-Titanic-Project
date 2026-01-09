
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')


def load_data():
    try:
        return pd.read_csv("data/train.csv")
    except FileNotFoundError:
        print("⚠ train.csv not found. Upload Kaggle dataset later.")
        return None


    
   
    data = {
        'PassengerId': range(1, n_samples + 1),
        'Pclass': np.random.choice([1, 2, 3], size=n_samples, p=[0.2, 0.2, 0.6]),
        'Name': [f"Passenger_{i}" for i in range(1, n_samples + 1)],
        'Sex': np.random.choice(['male', 'female'], size=n_samples, p=[0.65, 0.35]),
        'Age': np.concatenate([
            np.random.normal(35, 15, int(0.8 * n_samples)),  # Adults
            np.random.normal(8, 4, n_samples - int(0.8 * n_samples))  # Children
        ]),
        'SibSp': np.random.poisson(0.5, n_samples),
        'Parch': np.random.poisson(0.4, n_samples),
        'Ticket': [f"Ticket_{i}" for i in range(1, n_samples + 1)],
        'Fare': np.random.lognormal(3.5, 1.2, n_samples),
        'Cabin': [f"Cabin_{i}" if np.random.random() < 0.25 else "" for i in range(1, n_samples + 1)],
        'Embarked': np.random.choice(['S', 'C', 'Q'], size=n_samples, p=[0.7, 0.2, 0.1])
    }
    
    
    survival_prob = []
    for i in range(n_samples):
        base_prob = 0.4  
        
        if data['Sex'][i] == 'female':
            base_prob += 0.3
            
        
        if data['Age'][i] < 16:
            base_prob += 0.15
            
        
        if data['Pclass'][i] == 1:
            base_prob += 0.2
        elif data['Pclass'][i] == 2:
            base_prob += 0.1
            
        survival_prob.append(min(base_prob, 0.95)) 
    
    
    survived = [1 if np.random.random() < prob else 0 for prob in survival_prob]
    data['Survived'] = survived
    
    df = pd.DataFrame(data)
    
   
    age_missing_idx = np.random.choice(df.index, size=int(0.2 * n_samples), replace=False)
    df.loc[age_missing_idx, 'Age'] = np.nan
    
    embarked_missing_idx = np.random.choice(df.index, size=int(0.02 * n_samples), replace=False)
    df.loc[embarked_missing_idx, 'Embarked'] = np.nan
    
    return df


def perform_eda(df):
    """Perform comprehensive exploratory data analysis"""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nSurvival distribution:\n{df['Survived'].value_counts(normalize=True)}")
    
    # Summary statistics
    print(f"\nSummary statistics:\n{df.describe()}")
    
    # Feature-wise survival insights
    print("\nSurvival rate by feature:")
    print(f"By Sex:\n{df.groupby('Sex')['Survived'].mean()}")
    print(f"\nBy Pclass:\n{df.groupby('Pclass')['Survived'].mean()}")
    print(f"\nBy Embarked:\n{df.groupby('Embarked')['Survived'].mean()}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Survival distribution
    sns.countplot(data=df, x='Survived', ax=axes[0,0])
    axes[0,0].set_title('Survival Distribution')
    
    # Survival by gender
    sns.countplot(data=df, x='Sex', hue='Survived', ax=axes[0,1])
    axes[0,1].set_title('Survival by Gender')
    
    # Survival by class
    sns.countplot(data=df, x='Pclass', hue='Survived', ax=axes[0,2])
    axes[0,2].set_title('Survival by Class')
    
    # Age distribution
    df['Age'].hist(bins=30, ax=axes[1,0])
    axes[1,0].set_title('Age Distribution')
    
    # Fare distribution
    df['Fare'].hist(bins=30, ax=axes[1,1])
    axes[1,1].set_title('Fare Distribution')
    
    # Survival by embarked
    sns.countplot(data=df, x='Embarked', hue='Survived', ax=axes[1,2])
    axes[1,2].set_title('Survival by Embarked')
    
    plt.tight_layout()
    plt.show()


def preprocess_data(df):
    """Handle missing values, encode categorical variables, and create features"""
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 60)
    
    
    processed_df = df.copy()
    
   
    processed_df['Age'].fillna(processed_df['Age'].median(), inplace=True)
    processed_df['Embarked'].fillna(processed_df['Embarked'].mode()[0], inplace=True)
    processed_df['Fare'].fillna(processed_df['Fare'].median(), inplace=True)
    
    
    
    processed_df['FamilySize'] = processed_df['SibSp'] + processed_df['Parch'] + 1
    
    
    processed_df['IsAlone'] = (processed_df['FamilySize'] == 1).astype(int)
    
   
    processed_df['AgeGroup'] = pd.cut(processed_df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                                     labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    
    processed_df['FarePerPerson'] = processed_df['Fare'] / processed_df['FamilySize']
    
    
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    le_age_group = LabelEncoder()
    
    processed_df['Sex_encoded'] = le_sex.fit_transform(processed_df['Sex'])
    processed_df['Embarked_encoded'] = le_embarked.fit_transform(processed_df['Embarked'])
    processed_df['AgeGroup_encoded'] = le_age_group.fit_transform(processed_df['AgeGroup'].astype(str))
    
   
    encoders = {
        'Sex': le_sex,
        'Embarked': le_embarked,
        'AgeGroup': le_age_group
    }
    
    
    features_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId', 'Sex', 'Embarked', 'AgeGroup']
    processed_df = processed_df.drop(columns=features_to_drop)
    
    print(f"Processed dataset shape: {processed_df.shape}")
    print(f"Features: {list(processed_df.columns)}")
    
    return processed_df, encoders


def select_features(df):
    """Select meaningful features for the model"""
    print("\n" + "=" * 60)
    print("FEATURE SELECTION")
    print("=" * 60)
    
  
    feature_columns = [
        'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 
        'FamilySize', 'IsAlone', 'FarePerPerson',
        'Sex_encoded', 'Embarked_encoded', 'AgeGroup_encoded'
    ]
    
    X = df[feature_columns]
    y = df['Survived']
    
    print(f"Selected features: {feature_columns}")
    print(f"Feature matrix shape: {X.shape}")
    
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data using stratified sampling for class balance"""
    print("\n" + "=" * 60)
    print("TRAIN-TEST SPLIT")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"Training survival rate: {y_train.mean():.3f}")
    print(f"Test survival rate: {y_test.mean():.3f}")
    
    return X_train, X_test, y_train, y_test


def build_models(X_train, y_train):
    """Train multiple models with proper hyperparameters"""
    print("\n" + "=" * 60)
    print("MODEL BUILDING")
    print("=" * 60)
    
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
       
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return trained_models


def evaluate_models(trained_models, X_test, y_test):
    """Evaluate models using multiple metrics"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    results = {}
    
    for name, model in trained_models.items():
      
        y_pred = model.predict(X_test)
        
       
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
   
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    print(f"\nBest Model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")
    
    return results, best_model_name


def predict_survival(model, encoders, passenger_data):
    """
    Predict survival for a custom passenger profile
    passenger_data: dict with keys: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
    """
    print("\n" + "=" * 60)
    print("FINAL PREDICTION FOR CUSTOM PASSENGER")
    print("=" * 60)
    
   
    df = pd.DataFrame([passenger_data])
    
   
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                           labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    
    
    df['Sex_encoded'] = encoders['Sex'].transform(df['Sex'])
    df['Embarked_encoded'] = encoders['Embarked'].transform(df['Embarked'])
    df['AgeGroup_encoded'] = encoders['AgeGroup'].transform(df['AgeGroup'].astype(str))
    
   
    feature_columns = [
        'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 
        'FamilySize', 'IsAlone', 'FarePerPerson',
        'Sex_encoded', 'Embarked_encoded', 'AgeGroup_encoded'
    ]
    
    X = df[feature_columns]
    
    
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    print(f"Passenger Profile: {passenger_data}")
    print(f"Prediction: {'Survived' if prediction == 1 else 'Did Not Survive'}")
    print(f"Survival Probability: {probability[1]:.4f}")
    print(f"Death Probability: {probability[0]:.4f}")
    
    return prediction, probability


def main():
    """Execute the complete ML pipeline"""
    print("TITANIC SURVIVAL PREDICTION - PROFESSIONAL ML PIPELINE")
    print("=" * 60)
    
    
    df = load_data()
    print(f"Loaded dataset with {len(df)} records")
    
    
    perform_eda(df)
    
    processed_df, encoders = preprocess_data(df)
    
   
    X, y = select_features(processed_df)
    
    
    X_train, X_test, y_train, y_test = split_data(X, y)
 
    trained_models = build_models(X_train, y_train)
    
    results, best_model_name = evaluate_models(trained_models, X_test, y_test)
    
    best_model = results[best_model_name]['model']
    
    sample_passenger = {
        'Pclass': 3,
        'Sex': 'male',
        'Age': 22,
        'SibSp': 1,
        'Parch': 0,
        'Fare': 7.25,
        'Embarked': 'S'
    }
    
    prediction, probability = predict_survival(best_model, encoders, sample_passenger)
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED")
    print("=" * 60)
    
    return results, best_model_name, encoders


if __name__ == "__main__":
    results, best_model_name, encoders = main()
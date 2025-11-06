import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class FakeJobDetector:
    def __init__(self, dataset_path=None):
        """
        Initialize the Fake Job Detector model
        
        Args:
            dataset_path (str): Path to the CSV dataset file
        """
        self.pipeline = None
        self.dataset_path = r"C:\Users\choda\OneDrive\Desktop\lastone\fake_job_postings.csv"
        
        if dataset_path:
            # Load the dataset
            self.df = pd.read_csv("./fake_job_postings.csv")
            
            # Create comprehensive text feature
            self.df['comprehensive_text'] = (
                self.df['company_profile'].fillna('') + ' ' + 
                self.df['description'].fillna('') + ' ' + 
                self.df['requirements'].fillna('') + ' ' +
                self.df['job_id'].astype(str).fillna('') + ' ' +
                self.df['location'].fillna('') + ' ' +
                self.df['department'].fillna('') + ' ' +
                self.df['salary_range'].fillna('')
            )
            
            # Prepare features
            self._prepare_features()
            
            # Train the model
            self._train_model()
    
    def _prepare_features(self):
        """Prepare features for the machine learning model"""
        # Select features
        self.features = [
            'job_id', 'location', 'department', 'salary_range', 
            'has_company_logo', 'telecommuting', 'employment_type', 
            'comprehensive_text'
        ]
        
        # Preprocessing for numeric columns
        numeric_features = ['has_company_logo', 'telecommuting']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ])
        
        # Preprocessing for text features
        text_transformer = Pipeline(steps=[
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('text', text_transformer, 'comprehensive_text')
            ])
        
        # Prepare data
        self.X = self.df[self.features]
        self.y = self.df['fraudulent']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
    
    def _train_model(self):
        """Train the machine learning model"""
        # Create full pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200, 
                random_state=42, 
                class_weight='balanced',
                max_depth=10
            ))
        ])
        
        # Fit the pipeline
        self.pipeline.fit(self.X_train, self.y_train)
        
        # Calculate training accuracy
        train_accuracy = self.pipeline.score(self.X_train, self.y_train)
        test_accuracy = self.pipeline.score(self.X_test, self.y_test)
        
        print(f"Model trained successfully!")
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Testing accuracy: {test_accuracy:.4f}")
    
    def predict_job_post(self, job_description, additional_info=None):
        """
        Predict if a job post is fake or genuine
        
        Args:
            job_description (str): The job description text
            additional_info (dict): Additional job information
            
        Returns:
            dict: Prediction results
        """
        # Create a sample dataframe for prediction
        if additional_info is None:
            additional_info = {
                'job_id': 0,
                'location': 'unknown',
                'department': 'unknown',
                'salary_range': 'unknown',
                'has_company_logo': 0,
                'telecommuting': 0,
                'employment_type': 'unknown',
                'comprehensive_text': job_description
            }
        else:
            additional_info['comprehensive_text'] = job_description
        
        # Create DataFrame
        pred_df = pd.DataFrame([additional_info])
        
        # Predict
        prediction = self.pipeline.predict(pred_df)
        probability = self.pipeline.predict_proba(pred_df)
        
        # Create output
        return {
            'is_fake': bool(prediction[0]),
            'prediction_text': 'Fake' if prediction[0] == 1 else 'Real',
            'confidence': round(probability[0][prediction[0]] * 100, 2),
            'raw_probabilities': probability[0].tolist()
        }
    
    def save_model(self, filepath):
        """
        Save the trained model to a file
        
        Args:
            filepath (str): Path to save the model
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        print(f"Model saved successfully to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from a file
        
        Args:
            filepath (str): Path to the saved model
        """
        # Load the model
        with open(filepath, 'rb') as f:
            self.pipeline = pickle.load(f)
        
        print("Model loaded successfully from C:\\Users\\choda\\OneDrive\\Desktop\\lastone\\models\\fake_job_detector.pkl")


        
    def get_model_info(self):
        """Get information about the model and dataset"""
        if self.df is not None:
            return {
                'total_jobs': len(self.df),
                'fake_jobs': int(self.df['fraudulent'].sum()),
                'genuine_jobs': int(len(self.df) - self.df['fraudulent'].sum()),
                'fake_percentage': f"{(self.df['fraudulent'].sum() / len(self.df) * 100):.2f}%"
            }
        return {
            'status': 'Model loaded without dataset information'
        }

# Example usage when running the file directly
if __name__ == "__main__":
    # Path to your dataset
    dataset_path = r"C:\Users\choda\OneDrive\Desktop\lastone\fake_job_postings.csv"
    
    # Create and train model
    model = FakeJobDetector(r"C:\Users\choda\OneDrive\Desktop\lastone\fake_job_postings.csv")
    
    # Save the model
    model.save_model("models/fake_job_detector.pkl")
    
    # Test prediction
    test_job = """
    Job Title: Software Engineer
Department: Engineering
Location: Bengaluru, Karnataka
Reports To: Engineering Manager
Summary: As a Software Engineer, you will be responsible for designing, developing, testing, and maintaining software applications and systems, contributing to our team's success in building innovative solutions. 
You will work closely with other engineers, product managers, and designers to deliver high-quality software products.
    """
    
    result = model.predict_job_post(test_job)
    print(f"Prediction: {result['prediction_text']}")
    print(f"Confidence: {result['confidence']}%")
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import time
import warnings
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
import json
import os

from .preprocessing import Preprocessing, PreprocessingPretrained
from .embedding import TextEmbedder
# Add these imports with your existing ones
from .classification import (
    SklearnSentimentClassifier, 
    KerasSentimentClassifier,
    KNNSentimentClassifier,
)
warnings.filterwarnings('ignore')

class TextMiningPipeline:
    """
    Complete text mining pipeline that integrates preprocessing, embedding, and classification
    with comprehensive cross-validation and evaluation capabilities.
    """
    
    def __init__(self, 
                 # Preprocessing parameters
                 preprocessing_config: Optional[Dict[str, Any]] = None,
                 use_pretrained_preprocessing: bool = False,
                 
                 # Embedding parameters
                 embedding_config: Optional[Dict[str, Any]] = None,
                 
                 # Classification parameters
                 classification_config: Optional[Dict[str, Any]] = None,
                 
                 # Pipeline parameters
                 balance_dataset: bool = True,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize the text mining pipeline.
        
        Args:
            preprocessing_config (dict): Configuration for preprocessing
                {
                    'lemmatize': bool,
                    'stem': bool,
                    'emoji_support_level': int,
                    'translate': bool
                }
            use_pretrained_preprocessing (bool): Use lightweight preprocessing for pretrained models
            embedding_config (dict): Configuration for embeddings
                {
                    'method': str ('word2vec', 'bow', 'transformer'),
                    'model_name': str (for transformer),
                    'vector_size': int (for word2vec),
                    'max_features': int (for bow),
                    ... other method-specific parameters
                }
            classification_config (dict): Configuration for classification
                {
                    'type': str ('sklearn' or 'keras'),
                    'model_type': str (for sklearn),
                    'hidden_layers': list (for keras),
                    ... other model-specific parameters
                }
            balance_dataset (bool): Whether to balance the dataset using SMOTE-Tomek
            random_state (int): Random state for reproducibility
            verbose (bool): Whether to print detailed output
        """
        self.random_state = random_state
        self.verbose = verbose
        self.balance_dataset = balance_dataset
        
        # Set default configurations
        self.preprocessing_config = preprocessing_config or {
            'lemmatize': True,
            'stem': False,
            'emoji_support_level': 0,
            'translate': False
        }
        
        self.embedding_config = embedding_config or {
            'method': 'transformer',
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2'
        }
        
        self.classification_config = classification_config or {
            'type': 'sklearn',
            'model_type': 'logistic_regression',
            'C': 1.0,
            'max_iter': 1000
        }
        
        # Initialize components
        self.use_pretrained_preprocessing = True #use_pretrained_preprocessing
        self.preprocessor = None
        self.embedder = None
        self.classifier = None
        
        # Pipeline state
        self.is_fitted = False
        self.training_time = 0
        self.preprocessing_time = 0
        self.embedding_time = 0
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize preprocessing, embedding, and classification components."""
        try:
            # Initialize preprocessor
            if self.use_pretrained_preprocessing:
                self.preprocessor = PreprocessingPretrained(
                    translate=self.preprocessing_config.get('translate', False)
                )
            else:
                self.preprocessor = Preprocessing(**self.preprocessing_config)
            
            # Initialize embedder
            self.embedder = TextEmbedder(**self.embedding_config)
            
            # Initialize classifier
            classifier_type = self.classification_config['type']
            
            if classifier_type == 'sklearn':
                # Create a copy to avoid modifying the original config
                classifier_config = self.classification_config.copy()
                classifier_config.pop('type')
                
                self.classifier = SklearnSentimentClassifier(
                    random_state=self.random_state,
                    **classifier_config
                )
            
            elif classifier_type == 'keras':
                # Create a copy to avoid modifying the original config
                classifier_config = self.classification_config.copy()
                classifier_config.pop('type')
                
                self.classifier = KerasSentimentClassifier(
                    random_state=self.random_state,
                    **classifier_config
                )
            
            elif classifier_type == 'knn':
                # Create a copy to avoid modifying the original config
                classifier_config = self.classification_config.copy()
                classifier_config.pop('type')
                
                self.classifier = KNNSentimentClassifier(
                    random_state=self.random_state,
                    **classifier_config
                )
    
            
            else:
                raise ValueError(f"Unsupported classification type: {classifier_type}")
            
            if self.verbose:
                print(" Pipeline components initialized successfully")
                
        except Exception as e:
            print(f" Error initializing pipeline components: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    def set_input_dim(self, input_dim: int):
        """Set the input dimension for the LSTM model."""
        self.input_dim = input_dim
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess texts using the configured preprocessor.
        
        Args:
            texts (List[str]): Raw texts to preprocess
            
        Returns:
            List[str]: Preprocessed texts
        """
        if self.verbose:
            print(" Preprocessing texts...")
        
        start_time = time.time()
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Convert preprocessed tokens back to strings for embedding
        preprocessed = self.preprocessor.preprocess(texts)
        preprocessed_texts = [' '.join(tokens) if isinstance(tokens, list) else str(tokens) 
                            for tokens in preprocessed]
        
        self.preprocessing_time = time.time() - start_time
        
        if self.verbose:
            print(f" Preprocessing completed in {self.preprocessing_time:.2f}s")
            print(f"   Sample preprocessed text: {preprocessed_texts[0][:100]}...")
        
        return preprocessed_texts
    
    def generate_embeddings(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """
        Generate embeddings for preprocessed texts.
        
        Args:
            texts (List[str]): Preprocessed texts
            fit (bool): Whether to fit the embedder (for training)
            
        Returns:
            np.ndarray: Text embeddings
        """
        if self.verbose:
            print(f" Generating embeddings using {self.embedding_config.get('method', 'default')}...")
        
        start_time = time.time()
        
        try:
            # Special case for transformer classifiers that handle raw text
            if self.embedding_config.get('method') == 'raw_text':
                # For transformer classifiers, return the raw text as-is
                # The classifier will handle tokenization internally
                if self.verbose:
                    print(" Raw text passed through for transformer classifier")
                return texts  # Return texts directly, not as numpy array
            
            # Generate embeddings using the correct method names
            if fit:
                # Check if the embedder has fit_transform method
                if hasattr(self.embedder, 'fit_transform'):
                    embeddings = self.embedder.fit_transform(texts)
                else:
                    # Use separate fit and transform calls
                    self.embedder.fit(texts)
                    embeddings = self.embedder.transform(texts)
            else:
                embeddings = self.embedder.transform(texts)
            
            embedding_time = time.time() - start_time
        

            if hasattr(embeddings, 'shape') and embeddings.shape[1:]:
                self.embedding_dimension = embeddings.shape[1]
                if hasattr(self.classifier, 'set_input_dim'):
                    self.classifier.set_input_dim(self.embedding_dimension)
            
            if self.verbose:
                print(f" Embeddings generated in {embedding_time:.2f}s")
                if hasattr(embeddings, 'shape'):
                    print(f"   Embedding shape: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            print(f" Error generating embeddings: {str(e)}")
            raise

    
    def balance_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance the dataset using SMOTE-Tomek.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Balanced features and labels
        """
        if not self.balance_dataset:
            return X, y
        
        if self.verbose:
            print(" Balancing dataset using SMOTE-Tomek...")
            print(f"   Original distribution: {np.bincount(y)}")
        
        smote_tomek = SMOTETomek(random_state=self.random_state)
        X_balanced, y_balanced = smote_tomek.fit_resample(X, y)
        
        # Ensure non-negative values for certain classifiers
        X_balanced = np.clip(X_balanced, 1e-8, None)
        
        if self.verbose:
            print(f" Dataset balanced")
            print(f"   New distribution: {np.bincount(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def fit(self, texts: Union[List[str], pd.Series], labels: Union[List, pd.Series]):
        """
        Fit the entire pipeline on training data.
        
        Args:
            texts (Union[List[str], pd.Series]): Training texts
            labels (Union[List, pd.Series]): Training labels
        """
        if self.verbose:
            print(" Starting pipeline training...")
            print(f"   Training samples: {len(texts)}")
        
        start_time = time.time()
        
        # Step 1: Preprocess texts
        preprocessed_texts = self.preprocess_texts(texts)
        
        
        embeddings = self.generate_embeddings(preprocessed_texts, fit=True)
        
        # Step 3: Prepare labels
        if isinstance(labels, list):
            labels = pd.Series(labels)
        
        encoded_labels = self.classifier.prepare_labels(labels)
        
        # Step 4: Balance dataset if requested
        if self.balance_dataset:
            if self.verbose:
                print(" Balancing dataset...")
            # Balance the embeddings and labels
            embeddings, encoded_labels = self.balance_data(embeddings, encoded_labels)
        X_train, y_train = self.balance_data(embeddings, encoded_labels)
        
        # Step 5: Train classifier
        if self.verbose:
            print(" Training classifier...")
        
        classifier_start = time.time()
        
        if self.classification_config['type'] == 'keras':
            # Split for validation in Keras
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
            )
            
            self.classifier.train(
                X_train_split, y_train_split,
                X_val=X_val_split, y_val=y_val_split,
                epochs=self.classification_config.get('epochs', 50),
                batch_size=self.classification_config.get('batch_size', 32),
                verbose=0 if not self.verbose else 1
            )
        else:
            self.classifier.train(X_train, y_train)
        
        classifier_time = time.time() - classifier_start
        self.training_time = time.time() - start_time
        
        self.is_fitted = True
        
        if self.verbose:
            print(f" Pipeline training completed!")
            print(f"   Total time: {self.training_time:.2f}s")
            print(f"   - Preprocessing: {self.preprocessing_time:.2f}s")
            print(f"   - Embedding: {self.embedding_time:.2f}s")
            print(f"   - Classification: {classifier_time:.2f}s")
    
    def predict(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Make predictions on new texts.
        
        Args:
            texts (Union[List[str], pd.Series]): Texts to predict
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        
        # Preprocess and embed
        preprocessed_texts = self.preprocess_texts(texts)
        embeddings = self.generate_embeddings(preprocessed_texts, fit=False)
        
        # Predict
        return self.classifier.predict(embeddings)
    
    def predict_proba(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Get prediction probabilities for new texts.
        
        Args:
            texts (Union[List[str], pd.Series]): Texts to predict
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        
        # Preprocess and embed
        preprocessed_texts = self.preprocess_texts(texts)
        embeddings = self.generate_embeddings(preprocessed_texts, fit=False)
        
        # Predict probabilities
        return self.classifier.predict_proba(embeddings)
    
    def evaluate(self, texts: Union[List[str], pd.Series], labels: Union[List, pd.Series], 
                print_report: bool = True) -> Dict[str, Any]:
        """
        Evaluate the pipeline on test data.
        
        Args:
            texts (Union[List[str], pd.Series]): Test texts
            labels (Union[List, pd.Series]): Test labels
            print_report (bool): Whether to print the classification report
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before evaluation")
        
        if self.verbose:
            print("🔄 Evaluating pipeline...")
        
        # Preprocess and embed
        preprocessed_texts = self.preprocess_texts(texts)
        if self.embedding_config.get('method') != 'raw_text':
            # Generate embeddings
            embeddings = self.generate_embeddings(preprocessed_texts, fit=False)
        
            # Transform labels
            if isinstance(labels, list):
                labels = pd.Series(labels)
    
            encoded_labels = self.classifier.transform_labels(labels)
        
            # Evaluate
            results = self.classifier.evaluate(embeddings, encoded_labels, print_report=print_report)
        
            if self.verbose and not print_report:
                print(f" Evaluation completed - Accuracy: {results['accuracy']:.4f}")
        
            return results
        # For transformer classifiers with raw text, pass texts directly
        if self.verbose:
            print(" Using raw text for transformer classifier evaluation")
        return self.classifier.evaluate_raw_text(preprocessed_texts, labels, print_report=print_report)
        
    
    def cross_validate(self, texts: Union[List[str], pd.Series], labels: Union[List, pd.Series], 
                    cv: int = 5, **kwargs) -> Dict[str, float]:
        """
        Perform cross-validation on the entire pipeline.
        """
        if self.verbose:
            print(" Starting cross-validation...")
            print(f"   Folds: {cv}")
            print(f"   Samples: {len(texts)}")
        
        start_time = time.time()
        
        try:
            
            # Convert inputs to proper format and reset indices
            if isinstance(texts, pd.Series):
                texts = texts.reset_index(drop=True)
            if isinstance(labels, pd.Series):
                labels = labels.reset_index(drop=True)
            elif isinstance(labels, list):
                labels = pd.Series(labels)
            
            # Step 1: Preprocess all texts (skip for transformer classifiers with raw_text)
            if self.embedding_config.get('method') == 'raw_text':
                # For transformer classifiers, use original texts
                processed_texts = texts
            else:
                processed_texts = self.preprocess_texts(texts)
                processed_texts = self.generate_embeddings(processed_texts, fit=True)
            
            # Step 3: Prepare labels
            if isinstance(labels, list):
                labels = pd.Series(labels)
            encoded_labels = self.classifier.prepare_labels(labels)
                        
            
            cv_results = self.classifier.cross_validate(processed_texts, encoded_labels, cv=cv,  oversample=self.balance_dataset,
                                                        **kwargs)
            
            # Handle case where cv_results might be unexpected type
            if isinstance(cv_results, dict):
                result_dict = cv_results
            elif isinstance(cv_results, (list, tuple)) and len(cv_results) >= 6:
                result_dict = {
                    'avg_accuracy': float(cv_results[0]),
                    'std_accuracy': float(cv_results[1]),
                    'avg_macro_f1': float(cv_results[2]),
                    'std_macro_f1': float(cv_results[3]),
                    'avg_weighted_f1': float(cv_results[4]),
                    'std_weighted_f1': float(cv_results[5])
                }
                # Handle additional metrics if available (beyond the basic 6)
                if len(cv_results) > 6:
                    additional_keys = [
                        'avg_macro_precision', 'std_macro_precision', 'avg_macro_recall', 'std_macro_recall',
                        'avg_weighted_precision', 'std_weighted_precision', 'avg_weighted_recall', 'std_weighted_recall',
                        'min_avg_class_precision', 'max_avg_class_precision',
                        'min_avg_class_recall', 'max_avg_class_recall', 
                        'min_avg_class_f1', 'max_avg_class_f1'
                    ]
                    for i, key in enumerate(additional_keys):
                        if i + 6 < len(cv_results):
                            result_dict[key] = float(cv_results[i + 6])
            elif isinstance(cv_results, (int, float)):
                accuracy = float(cv_results)
                result_dict = {
                    'avg_accuracy': accuracy,
                    'std_accuracy': 0.0,
                    'avg_macro_f1': accuracy,
                    'std_macro_f1': 0.0,
                    'avg_weighted_f1': accuracy,
                    'std_weighted_f1': 0.0,
                    'min_avg_class_precision': accuracy,
                    'max_avg_class_precision': accuracy,
                    'min_avg_class_recall': accuracy,
                    'max_avg_class_recall': accuracy,
                    'min_avg_class_f1': accuracy,
                    'max_avg_class_f1': accuracy
                }
            else:
                print(f"  Unexpected cv_results format: {type(cv_results)}")
                result_dict = {
                    'avg_accuracy': 0.0,
                    'std_accuracy': 0.0,
                    'avg_macro_f1': 0.0,
                    'std_macro_f1': 0.0,
                    'avg_weighted_f1': 0.0,
                    'std_weighted_f1': 0.0,
                    'min_avg_class_precision': 0.0,
                    'max_avg_class_precision': 0.0,
                    'min_avg_class_recall': 0.0,
                    'max_avg_class_recall': 0.0,
                    'min_avg_class_f1': 0.0,
                    'max_avg_class_f1': 0.0
                }
            cv_time = time.time() - start_time
            
            if self.verbose:
                print(f" Cross-validation completed in {cv_time:.2f}s")
                print(" Results:")
                print(f"   Average Accuracy: {result_dict['avg_accuracy']:.4f} ± {result_dict['std_accuracy']:.4f}")
                print(f"   Average Macro F1: {result_dict['avg_macro_f1']:.4f} ± {result_dict['std_macro_f1']:.4f}")
                print(f"   Average Weighted F1: {result_dict['avg_weighted_f1']:.4f} ± {result_dict['std_weighted_f1']:.4f}")
                
                # Display min/max class performance if available
                if 'min_avg_class_f1' in result_dict and 'max_avg_class_f1' in result_dict:
                    print(f"   Class Performance Range:")
                    print(f"     F1-Score: {result_dict['min_avg_class_f1']:.4f} - {result_dict['max_avg_class_f1']:.4f}")
                    print(f"     Precision: {result_dict['min_avg_class_precision']:.4f} - {result_dict['max_avg_class_precision']:.4f}")
                    print(f"     Recall: {result_dict['min_avg_class_recall']:.4f} - {result_dict['max_avg_class_recall']:.4f}")            
            return result_dict
            
        except Exception as e:
            print(f" Error in cross_validate: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                'avg_accuracy': 0.0,
                'std_accuracy': 0.0,
                'avg_macro_f1': 0.0,
                'std_macro_f1': 0.0,
                'avg_weighted_f1': 0.0,
                'std_weighted_f1': 0.0,
                'min_avg_class_precision': 0.0,
                'max_avg_class_precision': 0.0,
                'min_avg_class_recall': 0.0,
                'max_avg_class_recall': 0.0,
                'min_avg_class_f1': 0.0,
                'max_avg_class_f1': 0.0
            }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the pipeline configuration and state.
        
        Returns:
            Dict[str, Any]: Pipeline information
        """
        return {
            'preprocessing_config': self.preprocessing_config,
            'use_pretrained_preprocessing': self.use_pretrained_preprocessing,
            'embedding_config': self.embedding_config,
            'classification_config': self.classification_config,
            'balance_dataset': self.balance_dataset,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted,
            'training_time': self.training_time,
            'preprocessing_time': self.preprocessing_time,
            'embedding_time': self.embedding_time,
            'classifier_info': self.classifier.get_model_info() if self.classifier else None
        }
    
    
    def load_pipeline(self, save_path: str):
        """
        Load a saved pipeline from disk.
        
        Args:
            save_path (str): Path to load the pipeline from (without extension)
        """
        # Load metadata
        with open(f"{save_path}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Restore configuration
        pipeline_info = metadata['pipeline_info']
        self.preprocessing_config = pipeline_info['preprocessing_config']
        self.embedding_config = pipeline_info['embedding_config']
        self.classification_config = pipeline_info['classification_config']
        self.balance_dataset = pipeline_info['balance_dataset']
        self.random_state = pipeline_info['random_state']
        
        # Reinitialize components
        self._initialize_components()
        
        # Load classifier
        self.classifier.load_model(f"{save_path}_classifier")
        
        # Note: Embedder needs to be refitted when loading
        self.is_fitted = False  # Will need to refit embedder
        
        if self.verbose:
            print(f" Pipeline loaded from {save_path}")
            print("  Note: You'll need to refit the embedder component")
    
    def __str__(self) -> str:
        """String representation of the pipeline."""
        return f"""TextMiningPipeline(
    Preprocessing: {self.preprocessing_config}
    Embedding: {self.embedding_config}
    Classification: {self.classification_config}
    Balanced: {self.balance_dataset}
    Fitted: {self.is_fitted}
)"""
    
    def __repr__(self) -> str:
        """Detailed representation of the pipeline."""
        return self.__str__()


# Utility functions for common pipeline configurations
def create_sklearn_pipeline(model_type: str = 'logistic_regression', 
                          embedding_method: str = 'transformer',
                          **kwargs) -> TextMiningPipeline:
    """
    Create a pre-configured sklearn-based pipeline.
    
    Args:
        model_type (str): Type of sklearn model
        embedding_method (str): Type of embedding method
        **kwargs: Additional configuration parameters
        
    Returns:
        TextMiningPipeline: Configured pipeline
    """
    classification_config = {
        'type': 'sklearn',
        'model_type': model_type,
        **kwargs.get('classification_params', {})
    }
    
    embedding_config = {
        'method': embedding_method,
        **kwargs.get('embedding_params', {})
    }
    
    return TextMiningPipeline(
        classification_config=classification_config,
        embedding_config=embedding_config,
        **kwargs.get('pipeline_params', {})
    )


def create_keras_pipeline(hidden_layers: List[int] = [128, 64],
                         embedding_method: str = 'transformer',
                         **kwargs) -> TextMiningPipeline:
    """
    Create a pre-configured Keras-based pipeline.
    
    Args:
        hidden_layers (List[int]): Hidden layer sizes
        embedding_method (str): Type of embedding method
        **kwargs: Additional configuration parameters
        
    Returns:
        TextMiningPipeline: Configured pipeline
    """
    classification_config = {
        'type': 'keras',
        'hidden_layers': hidden_layers,
        **kwargs.get('classification_params', {})
    }
    
    embedding_config = {
        'method': embedding_method,
        **kwargs.get('embedding_params', {})
    }
    
    return TextMiningPipeline(
        classification_config=classification_config,
        embedding_config=embedding_config,
        **kwargs.get('pipeline_params', {})
    )
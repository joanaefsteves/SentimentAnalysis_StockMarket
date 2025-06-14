# Group 35: 
# Joana Esteves, 20240746
# José Cavaco, 20240513 
# Leonardo Di Caterina 20240485
# Matilde Miguel, 20240549 
# Rita Serra, 20240515 

from abc import ABC, abstractmethod

import numpy as np #type: ignore
import joblib #type: ignore
from sklearn.preprocessing import LabelEncoder #type: ignore
from sklearn.metrics import accuracy_score, classification_report #type: ignore
from sklearn.model_selection import StratifiedKFold #type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.naive_bayes import MultinomialNB #type: ignore
from sklearn.tree import DecisionTreeClassifier #type: ignore
from sklearn.ensemble import RandomForestClassifier #type: ignore 
from sklearn.svm import SVC #type: ignore
import matplotlib.pyplot as plt #type: ignore

import tqdm #type: ignore
from tqdm import tqdm  #type: ignore
    

import joblib #type: ignore
import tensorflow as tf #type: ignore
from tensorflow import keras #type: ignore
from tensorflow.keras import layers #type: ignore

class BaseSentimentClassifier(ABC):
    """
    Abstract base class for sentiment classifiers
    Defines the common interface and shared functionality
    """
    
    def __init__(self, random_state=42):
        """
        Initialize base classifier
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.label_encoder = None
        self.is_trained = False
        self.num_classes = None
    
    def prepare_labels(self, target_column):
        """
        Convert ratings/labels to encoded sentiment labels
        
        Args:
            target_column (pandas.Series): Review ratings or labels
            
        Returns:
            numpy.ndarray: Encoded sentiment labels
        """
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        
        # Convert numeric ratings to sentiment labels if needed
        if target_column.dtype in ['int64', 'float64']:
            sentiment_labels = target_column.apply(
                lambda x: 'negative' if x == 0 else ('neutral' if x == 1 else 'positive')
            )
        else:
            sentiment_labels = target_column
        
        encoded_labels = self.label_encoder.fit_transform(sentiment_labels)
        self.num_classes = len(self.label_encoder.classes_)
        
        return encoded_labels
    
    def transform_labels(self, target_column):
        """
        Transform labels using existing label encoder (for validation/test sets)
        
        Args:
            target_column (pandas.Series): Review ratings or labels
            
        Returns:
            numpy.ndarray: Encoded sentiment labels
        """
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted. Call prepare_labels() first.")
        
        # Convert numeric ratings to sentiment labels if needed
        if target_column.dtype in ['int64', 'float64']:
            sentiment_labels = target_column.apply(
                lambda x: 'negative' if x == 0 else ('neutral' if x == 1 else 'positive')
            )
        else:
            sentiment_labels = target_column
        
        return self.label_encoder.transform(sentiment_labels)
    
    @abstractmethod
    def _build_model(self):
        """
        Build the specific model architecture
        Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        """
        Train the classifier
        Must be implemented by subclasses
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            **kwargs: Additional training parameters specific to each model
        """
        pass
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (numpy.ndarray): Features to predict
            
        Returns:
            numpy.ndarray: Predicted class labels
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        return self._predict_implementation(X)
    
    @abstractmethod
    def _predict_implementation(self, X):
        """
        Model-specific prediction implementation
        Must be implemented by subclasses
        """
        pass
    
    def predict_proba(self, X):
        """
        Get prediction probabilities (if supported)
        
        Args:
            X (numpy.ndarray): Features to predict
            
        Returns:
            numpy.ndarray: Prediction probabilities
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        return self._predict_proba_implementation(X)
    
    @abstractmethod
    def _predict_proba_implementation(self, X):
        """
        Model-specific probability prediction implementation
        Must be implemented by subclasses
        """
        pass
    
    def evaluate(self, X_test, y_test, print_report = False):
        """
        Evaluate model performance
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Use label names if available
        target_names = None
        if self.label_encoder is not None:
            target_names = self.label_encoder.classes_
        
        report = classification_report(y_test, predictions, target_names=target_names, output_dict=True)
        
        # Get additional metrics from subclass if available
        additional_metrics = self._get_additional_metrics(X_test, y_test)
        if print_report:
            print("Classification Report:\n", report)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': predictions
        }
        report.update(additional_metrics)
        return report
    
    def _get_additional_metrics(self, X_test, y_test):  
        return {}
    
    def cross_validate(self, X, y, cv=5, **kwargs):
        """
        Perform cross-validation
        
        Args:
            X (numpy.ndarray): Features
            y (numpy.ndarray): Labels
            cv (int): Number of cross-validation folds
            **kwargs: Additional parameters for training
            
        Returns:
            dict: Cross-validation scores
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = [] # list that will hold the classification report dictionaries of each fold
        tqdm.pandas(desc="Cross-validating")
        for fold, (train_idx, val_idx) in tqdm(enumerate(skf.split(X, y)), total=cv):
        
            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
            # Train and evaluate this fold
            fold_score = self._cross_validate_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold, fold, **kwargs)
            #print(f"Fold {fold + 1}/{cv} - Score: \n{fold_score}")
            scores.append(fold_score)
        
        # Aggregate results
        aggregated_scores = {}
        
        # Calculate average accuracy
        accuracies = [score['accuracy'] for score in scores]
        aggregated_scores['avg_accuracy'] = np.mean(accuracies)
        aggregated_scores['std_accuracy'] = np.std(accuracies)
        
        # Calculate averages for macro avg and weighted avg
        macro_precisions = [score['macro avg']['precision'] for score in scores]
        macro_recalls = [score['macro avg']['recall'] for score in scores]
        macro_f1s = [score['macro avg']['f1-score'] for score in scores]
        
        weighted_precisions = [score['weighted avg']['precision'] for score in scores]
        weighted_recalls = [score['weighted avg']['recall'] for score in scores]
        weighted_f1s = [score['weighted avg']['f1-score'] for score in scores]
        
        aggregated_scores.update({
            'avg_macro_precision': np.mean(macro_precisions),
            'std_macro_precision': np.std(macro_precisions),
            'avg_macro_recall': np.mean(macro_recalls),
            'std_macro_recall': np.std(macro_recalls),
            'avg_macro_f1': np.mean(macro_f1s),
            'std_macro_f1': np.std(macro_f1s),
            
            'avg_weighted_precision': np.mean(weighted_precisions),
            'std_weighted_precision': np.std(weighted_precisions),
            'avg_weighted_recall': np.mean(weighted_recalls),
            'std_weighted_recall': np.std(weighted_recalls),
            'avg_weighted_f1': np.mean(weighted_f1s),
            'std_weighted_f1': np.std(weighted_f1s)
        })
        
        # Calculate per-class averages (for classes '0', '1', '2', etc.)
        class_keys = [key for key in scores[0].keys() if key.isdigit()]
        
        for class_key in class_keys:
            class_precisions = [score[class_key]['precision'] for score in scores]
            class_recalls = [score[class_key]['recall'] for score in scores]
            class_f1s = [score[class_key]['f1-score'] for score in scores]
            
            aggregated_scores[f'avg_class_{class_key}_precision'] = np.mean(class_precisions)
            aggregated_scores[f'std_class_{class_key}_precision'] = np.std(class_precisions)
            aggregated_scores[f'avg_class_{class_key}_recall'] = np.mean(class_recalls)
            aggregated_scores[f'std_class_{class_key}_recall'] = np.std(class_recalls)
            aggregated_scores[f'avg_class_{class_key}_f1'] = np.mean(class_f1s)
            aggregated_scores[f'std_class_{class_key}_f1'] = np.std(class_f1s)
        
        #print("*" * 50)
        #print("Cross-Validation Results:")
        #print(f"Average Accuracy: {aggregated_scores['avg_accuracy']:.4f} ± {aggregated_scores['std_accuracy']:.4f}")
        #print(f"Average Macro F1: {aggregated_scores['avg_macro_f1']:.4f} ± {aggregated_scores['std_macro_f1']:.4f}")
        #print(f"Average Weighted F1: {aggregated_scores['avg_weighted_f1']:.4f} ± {aggregated_scores['std_weighted_f1']:.4f}")
        #print("*" * 50)

        return aggregated_scores
            

    
    @abstractmethod
    def _cross_validate_fold(self, X_train, y_train, X_val, y_val, fold, **kwargs) -> dict:
        """
        Train and evaluate a single cross-validation fold
        Must be implemented by subclasses
        
        Returns:
            float: Validation score for this fold
        """
        pass
    
    def save_model(self, model_path):
        """
        Save trained model and metadata
        
        Args:
            model_path (str): Path to save model (without extension)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Save common metadata
        metadata = {
            'label_encoder': self.label_encoder,
            'num_classes': self.num_classes,
            'random_state': self.random_state,
            'model_type': self.__class__.__name__
        }
        
        # Add model-specific metadata
        metadata.update(self._get_save_metadata())
        
        # Save metadata
        joblib.dump(metadata, f"{model_path}_metadata.pkl")
        
        # Save model using model-specific method
        self._save_model_implementation(model_path)
        
        print(f"Model saved to {model_path}")
    
    @abstractmethod
    def _save_model_implementation(self, model_path):
        """
        Model-specific save implementation
        Must be implemented by subclasses
        """
        pass
    
    def _get_save_metadata(self):
        """
        Get model-specific metadata for saving
        Can be overridden by subclasses
        
        Returns:
            dict: Model-specific metadata
        """
        return {}
    
    def load_model(self, model_path):
        """
        Load pre-trained model and metadata
        
        Args:
            model_path (str): Path to model file (without extension)
        """
        try:
            # Load metadata
            metadata = joblib.load(f"{model_path}_metadata.pkl")
            self.label_encoder = metadata['label_encoder']
            self.num_classes = metadata['num_classes']
            self.random_state = metadata['random_state']
            
            # Load model-specific metadata
            self._load_metadata(metadata)
            
            # Load model using model-specific method
            self._load_model_implementation(model_path)
            
            self.is_trained = True
            print(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.is_trained = False
    
    @abstractmethod
    def _load_model_implementation(self, model_path):
        """
        Model-specific load implementation
        Must be implemented by subclasses
        """
        pass
    
    def _load_metadata(self, metadata):
        """
        Load model-specific metadata
        Can be overridden by subclasses
        """
        pass
    
    def get_model_info(self):
        """
        Get information about the current model
        
        Returns:
            dict: Model information
        """
        base_info = {
            'model_class': self.__class__.__name__,
            'is_trained': self.is_trained,
            'random_state': self.random_state,
            'num_classes': self.num_classes,
            'has_label_encoder': self.label_encoder is not None,
            'label_classes': self.label_encoder.classes_.tolist() if self.label_encoder else None
        }
        
        # Add model-specific info
        specific_info = self._get_model_specific_info()
        base_info.update(specific_info)
        
        return base_info
    
    def _get_model_specific_info(self):
        """
        Get model-specific information
        Can be overridden by subclasses
        
        Returns:
            dict: Model-specific information
        """
        return {}
    
    def __str__(self):
        """String representation of the classifier"""
        return f"{self.__class__.__name__}(trained={self.is_trained}, classes={self.num_classes})"
    
    def __repr__(self):
        """Detailed string representation"""
        return self.__str__()
    


class SklearnSentimentClassifier(BaseSentimentClassifier):
    """
    Sklearn-based sentiment classifier implementation
    Supports multiple traditional ML algorithms
    """
    
    supported_classifiers = {
        'logistic_regression': LogisticRegression,
        'svm': SVC,
        'naive_bayes': MultinomialNB,
        'decision_tree': DecisionTreeClassifier,
        'random_forest': RandomForestClassifier
    }
    
    def __init__(self, model_type='logistic_regression', random_state=42, **model_params):
        """
        Initialize sklearn sentiment classifier
        
        Args:
            model_type (str): Type of classifier to use
            random_state (int): Random state for reproducibility
            **model_params: Additional parameters for the specific model
        """
        super().__init__(random_state=random_state)
        
        if model_type not in self.supported_classifiers:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {list(self.supported_classifiers.keys())}")
        
        self.model_type = model_type
        self.model_params = model_params
    
    def _build_model(self):
        """
        Build the sklearn model with appropriate parameters
        
        Returns:
            sklearn model: Configured model instance
        """
        # Set default parameters for each model type
        if self.model_type == 'svm':
            default_params = {'probability': True, 'random_state': self.random_state}
        elif self.model_type == 'naive_bayes':
            default_params = {}  # MultinomialNB doesn't use random_state
        else:
            default_params = {'random_state': self.random_state}
        
        # Merge with user-provided parameters
        final_params = {**default_params, **self.model_params}
        
        # Create and return the model
        model_class = self.supported_classifiers[self.model_type]
        return model_class(**final_params)
    
    def train(self, X_train, y_train, **kwargs):
        """
        Train the sklearn classifier
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            **kwargs: Additional parameters (ignored for sklearn models)
        """
        if self.is_trained:
            print("Warning: Model has already been trained. Creating new model instance.")
        
        # Build the model
        self.model = self._build_model()
        
        # Train the model
        try:
            self.model.fit(X_train, y_train)
            self.is_trained = True
            print(f"Sklearn {self.model_type} model trained successfully.")
        except Exception as e:
            print(f"Error training model: {e}")
            raise
        
        return self.model
    
    def _predict_implementation(self, X):
        """
        Sklearn-specific prediction implementation
        
        Args:
            X (numpy.ndarray): Features to predict
            
        Returns:
            numpy.ndarray: Predictions
        """
        return self.model.predict(X)
    
    def _predict_proba_implementation(self, X):
        """
        Sklearn-specific probability prediction implementation
        
        Args:
            X (numpy.ndarray): Features to predict
            
        Returns:
            numpy.ndarray: Prediction probabilities
        """
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"Model {self.model_type} does not support probability predictions.")
        
        return self.model.predict_proba(X)
    
    def _cross_validate_fold(self, X_train, y_train, X_val, y_val, fold, **kwargs):
        """
        Train and evaluate a single cross-validation fold for sklearn models
        
        Args:
            X_train (numpy.ndarray): Training features for this fold
            y_train (numpy.ndarray): Training labels for this fold
            X_val (numpy.ndarray): Validation features for this fold
            y_val (numpy.ndarray): Validation labels for this fold
            fold (int): Fold number (for logging)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            float: Validation accuracy for this fold
        """
        # Create and train model for this fold
        fold_model = self._build_model()
        fold_model.fit(X_train, y_train)
        
        # Evaluate on validation set
        predictions = fold_model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        # Fixed: Changed variable name to avoid shadowing the imported function
        report_accuracy = classification_report(y_val, predictions, output_dict=True)#['accuracy']
        
        return report_accuracy
    
    def _save_model_implementation(self, model_path):
        """
        Save the sklearn model
        
        Args:
            model_path (str): Path to save model (without extension)
        """
        joblib.dump(self.model, f"{model_path}_sklearn_model.pkl")
    
    def _get_save_metadata(self):
        """
        Get sklearn-specific metadata for saving
        
        Returns:
            dict: Sklearn-specific metadata
        """
        return {
            'model_type': self.model_type,
            'model_params': self.model_params
        }
    
    def _load_model_implementation(self, model_path):
        """
        Load the sklearn model
        
        Args:
            model_path (str): Path to model file (without extension)
        """
        self.model = joblib.load(f"{model_path}_sklearn_model.pkl")
    
    def _load_metadata(self, metadata):
        """
        Load sklearn-specific metadata
        
        Args:
            metadata (dict): Loaded metadata dictionary
        """
        self.model_type = metadata['model_type']
        self.model_params = metadata['model_params']
    
    def get_feature_importance(self):
        """
        Get feature importance for tree-based models
        
        Returns:
            numpy.ndarray: Feature importance scores
            
        Raises:
            ValueError: If model doesn't support feature importance
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, return absolute coefficients
            coef = self.model.coef_
            if len(coef.shape) > 1:
                # Multi-class case - return mean absolute coefficients
                return np.mean(np.abs(coef), axis=0)
            else:
                # Binary case
                return np.abs(coef.flatten())
        else:
            raise ValueError(f"Model {self.model_type} does not support feature importance.")
    
    def _get_model_specific_info(self):
        """
        Get sklearn-specific model information
        
        Returns:
            dict: Sklearn-specific information
        """
        info = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'supports_probability': hasattr(self.model, 'predict_proba') if self.model else False,
            'supports_feature_importance': (
                hasattr(self.model, 'feature_importances_') or 
                hasattr(self.model, 'coef_')
            ) if self.model else False
        }
        
        # Add model-specific parameters if trained
        if self.is_trained and self.model:
            info['model_parameters'] = self.model.get_params()
            
            # Add number of features if available
            if hasattr(self.model, 'n_features_in_'):
                info['n_features'] = self.model.n_features_in_
        
        return info
    
    def get_model_coefficients(self):
        """
        Get model coefficients for linear models
        
        Returns:
            numpy.ndarray: Model coefficients
            
        Raises:
            ValueError: If model doesn't have coefficients
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        if hasattr(self.model, 'coef_'):
            return self.model.coef_
        else:
            raise ValueError(f"Model {self.model_type} does not have coefficients.")
    
    def get_support_vectors(self):
        """
        Get support vectors for SVM models
        
        Returns:
            numpy.ndarray: Support vectors
            
        Raises:
            ValueError: If model is not SVM
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        if self.model_type == 'svm' and hasattr(self.model, 'support_vectors_'):
            return self.model.support_vectors_
        else:
            raise ValueError("Support vectors are only available for SVM models.")
    
    @classmethod
    def get_supported_models(cls):
        """
        Get list of supported model types
        
        Returns:
            list: List of supported model type strings
        """
        return list(cls.supported_classifiers.keys())
    
    def __str__(self):
        """String representation of the sklearn classifier"""
        return f"SklearnSentimentClassifier(model_type='{self.model_type}', trained={self.is_trained})"




class KerasSentimentClassifier(BaseSentimentClassifier):
    """
    Keras Neural Network-based sentiment classifier implementation
    """
    
    def __init__(self, hidden_layers=[128, 64], dropout_rate=0.3, learning_rate=0.001, 
                 activation='relu', optimizer='adam', random_state=42):
        """
        Initialize Keras sentiment classifier
        
        Args:
            hidden_layers (list): List of hidden layer sizes
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
            activation (str): Activation function for hidden layers
            optimizer (str or keras.optimizers): Optimizer to use
            random_state (int): Random state for reproducibility
        """
        super().__init__(random_state=random_state)
        
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.input_dim = None
        self.history = None
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
    
    def _build_model(self):
        """
        Build the neural network architecture

        Returns:
            keras.Model: Compiled Keras model
        """
        if self.input_dim is None:
            raise ValueError("input_dim must be set before building the model")

        model = keras.Sequential([
            keras.layers.Input(shape=(self.input_dim,)),
            keras.layers.BatchNormalization()
        ])

        # Add hidden layers
        for i, units in enumerate(self.hidden_layers):
            model.add(keras.layers.Dense(units, activation=self.activation, name=f'dense_{i+1}'))
            model.add(keras.layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}'))
            if i < len(self.hidden_layers) - 1:  # Don't add batch norm after last hidden layer
                model.add(keras.layers.BatchNormalization(name=f'batch_norm_{i+1}'))

        # Output layer
        if self.num_classes == 2:
            model.add(keras.layers.Dense(1, activation='sigmoid', name='output'))
            loss = 'binary_crossentropy'
        else:
            model.add(keras.layers.Dense(self.num_classes, activation='softmax', name='output'))
            loss = 'sparse_categorical_crossentropy'

        # Configure optimizer
        if isinstance(self.optimizer, str):
            if self.optimizer == 'adam':
                optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
            elif self.optimizer == 'sgd':
                optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)
            elif self.optimizer == 'rmsprop':
                optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer string: {self.optimizer}")
        else:
            optimizer = self.optimizer

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        return model

    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, 
              early_stopping=True, patience=10, verbose=1, **kwargs):
        """
        Train the neural network
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            X_val (numpy.ndarray): Validation features (optional)
            y_val (numpy.ndarray): Validation labels (optional)
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            early_stopping (bool): Whether to use early stopping
            patience (int): Early stopping patience
            verbose (int): Verbosity level
            **kwargs: Additional training parameters
        """
        if self.is_trained:
            print("Warning: Model has already been trained. Creating new model instance.")
        
        # Set input dimension and number of classes
        if self.input_dim is None:
            self.input_dim = X_train.shape[1]
        
        if self.num_classes is None:
            self.num_classes = len(np.unique(y_train))
        
        # Build model
        self.model = self._build_model()
        
        if verbose:
            print(f"Neural Network Architecture:")
            self.model.summary()
        
        # Prepare callbacks
        callbacks = []
        
        if early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=verbose if verbose > 0 else 0
            )
            callbacks.append(early_stop)
        
        # Reduce learning rate on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=max(1, patience//2),
            min_lr=1e-6,
            verbose=verbose if verbose > 0 else 0
        )
        callbacks.append(reduce_lr)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train the model
        try:
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=verbose
            )
            self.is_trained = True
            actual_epochs = len(self.history.history['loss'])
            print(f"Neural network trained successfully for {actual_epochs} epochs.")
            
        except Exception as e:
            print(f"Error training model: {e}")
            raise
        
        return self.history
    
    def _predict_implementation(self, X):
        """
        Keras-specific prediction implementation
        
        Args:
            X (numpy.ndarray): Features to predict
            
        Returns:
            numpy.ndarray: Predicted class labels
        """
        probabilities = self.model.predict(X, verbose=0)
        
        if self.num_classes == 2:
            # Binary classification
            predictions = (probabilities > 0.5).astype(int).flatten()
        else:
            # Multi-class classification
            predictions = np.argmax(probabilities, axis=1)
        
        return predictions
    
    def _predict_proba_implementation(self, X):
        """
        Keras-specific probability prediction implementation
        
        Args:
            X (numpy.ndarray): Features to predict
            
        Returns:
            numpy.ndarray: Prediction probabilities
        """
        probabilities = self.model.predict(X, verbose=0)
        
        if self.num_classes == 2:
            # For binary classification, return probabilities for both classes
            probabilities = np.column_stack([1 - probabilities, probabilities])
        
        return probabilities
    
    def _get_additional_metrics(self, X_test, y_test):
        """
        Get additional Keras-specific metrics
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
            
        Returns:
            dict: Additional metrics including test loss
        """
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
    
    def _cross_validate_fold(self, X_train, y_train, X_val, y_val, fold, epochs=50, 
                           batch_size=32, **kwargs):
        """
        Train and evaluate a single cross-validation fold for Keras models
        
        Args:
            X_train (numpy.ndarray): Training features for this fold
            y_train (numpy.ndarray): Training labels for this fold
            X_val (numpy.ndarray): Validation features for this fold
            y_val (numpy.ndarray): Validation labels for this fold
            fold (int): Fold number (for logging)
            epochs (int): Number of epochs for this fold
            batch_size (int): Batch size for training
            **kwargs: Additional parameters
            
        Returns:
            float: Validation accuracy for this fold
        """
        # Create model for this fold
        if self.input_dim is None:
            self.input_dim = X_train.shape[1]
        if self.num_classes is None:
            self.num_classes = len(np.unique(y_train))
        
        fold_model = self._build_model()
        
        # Train with early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=0
        )
        
        fold_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        _, accuracy = fold_model.evaluate(X_val, y_val, verbose=0)
        
        # Clean up to prevent memory issues
        del fold_model
        tf.keras.backend.clear_session()
        
        return accuracy
    
    def _save_model_implementation(self, model_path):
        """
        Save the Keras model
        
        Args:
            model_path (str): Path to save model (without extension)
        """
        self.model.save(f"{model_path}_keras_model.h5")
    
    def _get_save_metadata(self):
        """
        Get Keras-specific metadata for saving
        
        Returns:
            dict: Keras-specific metadata
        """
        return {
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'activation': self.activation,
            'optimizer': self.optimizer if isinstance(self.optimizer, str) else 'custom',
            'input_dim': self.input_dim
        }
    
    def _load_model_implementation(self, model_path):
        """
        Load the Keras model
        
        Args:
            model_path (str): Path to model file (without extension)
        """
        self.model = keras.models.load_model(f"{model_path}_keras_model.h5")
    
    def _load_metadata(self, metadata):
        """
        Load Keras-specific metadata
        
        Args:
            metadata (dict): Loaded metadata dictionary
        """
        self.hidden_layers = metadata['hidden_layers']
        self.dropout_rate = metadata['dropout_rate']
        self.learning_rate = metadata['learning_rate']
        self.activation = metadata['activation']
        self.optimizer = metadata['optimizer']
        self.input_dim = metadata['input_dim']
    
    def plot_training_history(self):
        """
        Plot training history (requires matplotlib)
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        try:
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot loss
            ax1.plot(self.history.history['loss'], label='Train Loss')
            if 'val_loss' in self.history.history:
                ax1.plot(self.history.history['val_loss'], label='Val Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot accuracy
            ax2.plot(self.history.history['accuracy'], label='Train Accuracy')
            if 'val_accuracy' in self.history.history:
                ax2.plot(self.history.history['val_accuracy'], label='Val Accuracy')
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Cannot plot training history.")
            print("Install matplotlib with: pip install matplotlib")
    
    def _get_model_specific_info(self):
        """
        Get Keras-specific model information
        
        Returns:
            dict: Keras-specific information
        """
        info = {
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'activation': self.activation,
            'optimizer': self.optimizer if isinstance(self.optimizer, str) else 'custom',
            'input_dim': self.input_dim,
            'total_parameters': self.model.count_params() if self.model else None,
            'has_training_history': self.history is not None
        }
        
        if self.history is not None:
            info['training_epochs'] = len(self.history.history['loss'])
            info['final_train_loss'] = self.history.history['loss'][-1]
            info['final_train_accuracy'] = self.history.history['accuracy'][-1]
            
            if 'val_loss' in self.history.history:
                info['final_val_loss'] = self.history.history['val_loss'][-1]
                info['final_val_accuracy'] = self.history.history['val_accuracy'][-1]
        
        return info
    
    def get_layer_weights(self, layer_name=None):
        """
        Get weights from a specific layer or all layers
        
        Args:
            layer_name (str): Name of layer to get weights from (optional)
            
        Returns:
            dict or numpy.ndarray: Layer weights
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        if layer_name:
            try:
                layer = self.model.get_layer(layer_name)
                return layer.get_weights()
            except ValueError:
                raise ValueError(f"Layer '{layer_name}' not found in model.")
        else:
            # Return all layer weights
            weights = {}
            for layer in self.model.layers:
                if layer.get_weights():  # Only include layers with weights
                    weights[layer.name] = layer.get_weights()
            return weights
    
    def __str__(self):
        """String representation of the Keras classifier"""
        layers_str = f"hidden_layers={self.hidden_layers}"
        return f"KerasSentimentClassifier({layers_str}, trained={self.is_trained})"
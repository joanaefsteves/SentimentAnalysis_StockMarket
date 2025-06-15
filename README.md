# TM Project: Predicting market behavior from tweets

Report link: https://1drv.ms/w/c/cf1a49b9f4993659/ERgIsTTHHINKgr_a_xW7wUEBcJEBMgYH1T6zx2J2EXEQXg?e=2Ng7z2

This project was developed with the goal of building an NLP model capable of predicting financial market sentiment based on tweets.

## 📁 Repository Structure
- `Data/` folder containing the CSV files with the provided training and test datasets.
- `Handout/` folder containing the project assignment and related materials.
- `Notebooks/` folder including all the notebooks used to explore, test, and develop the model:
  - `0_EDA.ipynb` performs exploratory data analysis.
  - `1_test_preprocessing.ipynb` contains tests for the developed preprocessing methods.
  - `2_test_embedding.ipynb` explores different embedding techniques.
  - `3_test_classifier.ipynb` includes tests with traditional classifiers.
  - `4_gridsearch_pipeline.ipynb` builds and tunes pipelines using both classical machine learning models (e.g., Logistic Regression, Random Forest) and deep learning models with Keras, optimizing their performance via grid search.
  - `5_transformer_encoders.ipynb` fine-tunes and evaluates transformer-based models (e.g., FinBERT, BERTweet) using Hugging Face’s Trainer API for sequence classification.
  - `6_final_model_g35.ipynb` integrates the entire pipeline with the best model and final configuration.
- `src/` folder contains reusable modules developed for the project:
  - `__init__.py` module initialization file.
  - `preprocessing.py` implements cleaning and text transformation methods.
  - `embedding.py` provides embedding generation methods using BoW, Word2Vec, and Transformers.
  - `classification.py` contains the logic for training and evaluating the classifiers.
  - `pipeline.py` implements the end-to-end prediction pipeline.
  - `tranformer_encoder.py` contains the implementation of transformer-based models.

## 🔧 Preprocessing

Two preprocessing classes were developed in `preprocessing` module:

- `Preprocessing`: applies cleaning, lemmatization, stemming, regex-based normalization, noise treatment (hashtags, URLs, etc.), and translation of non-English tweets.
- `PreprocessingPretrained`: a simplified version designed for Transformer-based models, applying only emoji demojization and translation.

## 📝 Classification Models

### 1. **Classical Machine Learning Models**
Implemented using `scikit-learn`:
- Logistic Regression  
- Random Forest  
- Support Vector Machines (LinearSVC, SVM with RBF)  
- K-Nearest Neighbors

### 2. **Deep Learning Models**
Two types of deep learning approaches were explored:
- **Feedforward Neural Networks**: Built with `Keras` and integrated into `GridSearchCV` using `KerasClassifier`. These models were included in the comparison alongside classical models.
- **Transformer-Based Models**: Fine-tuned using the Hugging Face `Trainer` API:
  - FinBERT
  - BERTweet
    
## 📊 Evaluation

The following metrics were used to assess model performance:

- **Accuracy**
- **Precision** (macro and weighted)
- **Recall** (macro and weighted)
- **F1-Score** (macro and weighted)



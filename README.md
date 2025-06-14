# TM Project: Predicting market behavior from tweets

Report link: https://1drv.ms/w/c/cf1a49b9f4993659/ERgIsTTHHINKgr_a_xW7wUEBcJEBMgYH1T6zx2J2EXEQXg?e=2Ng7z2

This project was developed with the goal of building an NLP model capable of predicting financial market sentiment based on tweets.

## 📁 Repository Structure
- `Data/` folder containing the CSV files with the provided training and test datasets.
- `Notebooks/` folder including all the notebooks used to explore, test, and develop the model:
  - `EDA.ipynb` performs exploratory data analysis.
  - `test_preprocessing.ipynb` contains tests for the developed preprocessing methods.
  - `test_classifier.ipynb` includes tests with traditional classifiers.
  - `transformer_encoder.ipynb` implements and trains the FinBERT and BERTweet models.
  - `full_pipeline.ipynb` integrates all steps of the final pipeline into a single flow.
- `src/` folder contains reusable modules developed for the project:
  - `preprocessing.py` implements cleaning and text transformation methods.
  - `embedding.py` provides embedding generation methods using BoW, Word2Vec, and Transformers (SentenceTransformers).
  - `classification.py` contains the logic for training and evaluating the classifiers.

## 🔧 Preprocessing

Two preprocessing classes were developed in `preprocessing` module:

- `Preprocessing`: applies cleaning, lemmatization, stemming, regex-based normalization, noise treatment (hashtags, URLs, etc.), and translation of non-English tweets.
- `PreprocessingPretrained`: a simplified version designed for Transformer-based models, applying only emoji demojization and translation.

## 📝 Classification Models
...

## 📊 Evaluation

The following metrics were used to assess model performance:

- **Accuracy**
- **Precision** (macro and weighted)
- **Recall** (macro and weighted)
- **F1-Score** (macro and weighted)



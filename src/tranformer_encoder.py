# Group 35: 
# Joana Esteves, 20240746
# José Cavaco, 20240513 
# Leonardo Di Caterina 20240485
# Matilde Miguel, 20240549 
# Rita Serra, 20240515 

# General
import numpy as np
import pandas as pd

# Model
import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, EarlyStoppingCallback, Trainer
from datasets import Dataset, DatasetDict

# Imbalance
from sklearn.utils.class_weight import compute_class_weight

# Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report

class TransformerEncoder():
    """
    Abstract base class for sentiment classifiers
    Defines the common interface and shared functionality
    """
    
    def __init__(self, num_classes, model_name, base_model="BERT", batch_size=16, learning_rate=3e-5, num_epochs=10, random_state=42, use_wandb=True):
        """
        Initialize base classifier
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.model_name = model_name
        self.base_model = base_model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.use_wandb = use_wandb 
        self.label2id = None
        self.id2label = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def train_predict(self, X_train, y_train, X_val, y_val, X_test=None):

        datasets = self._prepare_data(X_train, y_train, X_val, y_val, X_test)

        self._prepare_labels()

        class_weights = self._class_weights(datasets["train"]["label"])

        tokenizer, tokenized_datasets = self._tokenize(datasets)

        TrainerClass = self._trainer()

        trainer = self._build_model(TrainerClass, tokenizer, tokenized_datasets, class_weights)

        # Train
        trainer.train()

        # Predict
        if 'test' in tokenized_datasets:
            predictions = trainer.predict(tokenized_datasets['test'])
            preds = np.argmax(predictions.predictions, axis=1)  

            test_ids = tokenized_datasets['test']['id']

            df_preds = pd.DataFrame({
                'id': test_ids,
                'prediction': preds
            })
            df_preds.to_csv('../pred_35.csv', index=False)
            report = None
        else:
            predictions = trainer.predict(tokenized_datasets['val'])
            # Evaluate 
            preds = np.argmax(predictions.predictions, axis=1)
            labels = predictions.label_ids
            report = classification_report(labels, preds, target_names=["bearish", "bullish", "neutral"])
    
        return preds, report

    def _trainer(self):

        # Custom trainer to account for imbalance
        class WeightedLossTrainer(Trainer):
            def __init__(self, *args, class_weights=None, **kwargs):
                super().__init__(*args, **kwargs)
                
                if hasattr(self.model, 'device'):
                    self.device = self.model.device
                else:
                    if torch.cuda.is_available():
                        self.device = torch.device("cuda")
                    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                        self.device = torch.device("mps")
                    else:
                        self.device = torch.device("cpu")

                if class_weights is not None:
                    if isinstance(class_weights, torch.Tensor):
                        self.class_weights = class_weights.to(self.device)
                    else:
                        self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
                else:
                    self.class_weights = None

                self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                loss = self.loss_fct(logits, labels)
                return (loss, outputs) if return_outputs else loss

        return WeightedLossTrainer
       

    def _build_model(self, TrainerClass, tokenizer, tokenized_datasets, class_weights):

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id
        )

        if self.base_model == "BERT":
            # Freeze body
            for param in model.bert.parameters():
                param.requires_grad = False
        elif self.base_model == "ROBERTA":
            # Freeze body
            for param in model.roberta.parameters():
                param.requires_grad = False

        # Define training arguments for fine-tuning 
        training_args = TrainingArguments(
            # default optimizer = adam
            output_dir=f"../logs/{self.model_name}",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir='./logs',
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False, # we are minimizing loss
            report_to="wandb" if self.use_wandb else "none"
        )

        Trainer = TrainerClass

        # Initialize custom trainer with weighted loss to handle class imbalance
        model = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['val'],
            tokenizer=tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            class_weights=class_weights
        )

        return model
    
    def _tokenize(self, datasets):

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenized_datasets = datasets.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128), batched=True)

        return tokenizer, tokenized_datasets
    
    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average="macro"),
            "recall": recall_score(labels, preds, average="macro"),
            "f1": f1_score(labels, preds, average="macro"),
        }
    
    def _prepare_data(self, X_train, y_train, X_val, y_val, X_test=None):
        
        # Create DatasetDict - format needed for pre trained model 
        train_df = pd.DataFrame({'text': X_train, 'label': y_train})
        val_df = pd.DataFrame({'text': X_val, 'label': y_val})

        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)

        datasets = DatasetDict({
            'train': train_dataset,
            'val': val_dataset,
        })

        if X_test is not None:
            test_df = pd.DataFrame({'text': X_test['text'], 'id': X_test['id']})
            test_dataset = Dataset.from_pandas(test_df)
            datasets['test'] = test_dataset

        return datasets 
    
    def _prepare_labels(self):

        self.label2id = {"bearish": 0, "bullish": 1, "neutral": 2}
        self.id2label = {0: "bearish", 1: "bullish", 2: "neutral"}

    def _class_weights(self, y_train):

        class_weights_np = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(self.device)

        return class_weights
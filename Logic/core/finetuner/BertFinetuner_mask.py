import json
import pandas as pd
import numpy as np
import torch

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction
from sklearn.model_selection import train_test_split

class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        self.file_path = file_path
        self.top_n_genres = 5

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        data = None
        with open(self.file_path) as f:
            data = json.load(f)
        df = pd.json_normalize(data)
        df = df[['id', 'first_page_summary', 'genres']]
        df['first_page_summary'] = df['first_page_summary'].astype('string')
        exploded_df = df.explode('genres')
        one_hot_df = pd.get_dummies(exploded_df, columns=['genres'], prefix='', prefix_sep='')
        self.df = one_hot_df.groupby('id').max().reset_index().astype(int, errors = 'ignore').dropna()
        
        print('Head of loaded  dataset:')
        print(df.head())

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        cnts = {}
        df = self.df
        for col in df:
            s = sum([x == 1 for x in df[col]])
            if type(s) == int:
                cnts[col] = s
        cnts = {k: v for k, v in sorted(cnts.items(), key=lambda item: item[1], reverse=True)}
        genres = list(cnts.keys())[:self.top_n_genres]
        print('Top genres are:', ' '.join(genres))
        self.df = df[['first_page_summary', 'id'] + genres]
        print(self.df.head())

        self.label2id = {genre: idx for idx, genre in enumerate(genres)}
        self.id2label = {idx: genre for genre, idx in self.label2id.items()}

    def split_dataset(self, test_size=0.3, val_size=0.5):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        train_df, test_df = train_test_split(
              self.df,
              test_size=test_size,
              )
        print(f"Number of rows in training set: {len(train_df)}")
        print(f"Number of rows in test set: {len(test_df)}")

        not_chosen_columns = ['id', 'first_page_summary']

        label_columns = [col for col in self.df.columns if col not in not_chosen_columns]

        df_labels_train = train_df[label_columns]
        df_labels_test = test_df[label_columns]

        labels_list_train = df_labels_train.values.tolist()
        labels_list_test = df_labels_test.values.tolist()

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        train_texts = train_df['first_page_summary'].tolist()
        self.train_labels = labels_list_train

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        eval_texts = test_df['first_page_summary'].tolist()
        self.eval_labels = labels_list_test

        self.train_encodings = tokenizer(text=train_texts, padding="max_length", truncation=True, max_length=64)
        self.eval_encodings = tokenizer(text=eval_texts, padding="max_length", truncation=True, max_length=64)

        

    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        return IMDbDataset(encodings, labels)

    def fine_tune_bert(self, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            problem_type="multi_label_classification",
            num_labels=self.top_n_genres,
            id2label=self.id2label,
            label2id=self.label2id,
            )
        training_arguments = TrainingArguments(
            output_dir=".",
            eval_strategy="epoch",
            save_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps, 
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            report_to='none',# to turn off Wandb :)
            )
        
        train_dataset = self.create_dataset(self.train_encodings, self.train_labels)
        eval_dataset = self.create_dataset(self.eval_encodings, self.eval_labels)
        
        self.trainer = Trainer(
            model=self.model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            )

        self.trainer.train()

    def multi_label_metrics(self, predictions, labels, threshold=0.5):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'f1': f1_micro_average,
                   'roc_auc': roc_auc,
                   'accuracy': accuracy,
                   }
        return metrics
    def compute_metrics(self, pred: EvalPrediction):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        result = self.multi_label_metrics(
            predictions=preds, 
            labels=pred.label_ids,
            )
        return result

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        return self.trainer.evaluate()

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        self.model.save_pretrained(model_name)

class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float) 

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)
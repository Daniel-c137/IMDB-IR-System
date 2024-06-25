import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data_loader import ReviewLoader
from .basic_classifier import BasicClassifier


class ReviewDataSet(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

        if len(self.embeddings) != len(self.labels):
            raise Exception("Embddings and Labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.embeddings[i], self.labels[i]


class MLPModel(nn.Module):
    def __init__(self, in_features=100, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, xb):
        return self.network(xb)


class DeepModelClassifier(BasicClassifier):
    def __init__(self, in_features, num_classes, batch_size, num_epochs=50):
        """
        Initialize the model with the given in_features and num_classes
        Parameters
        ----------
        in_features: int
            The number of input features
        num_classes: int
            The number of classes
        batch_size: int
            The batch size of dataloader
        """
        super().__init__()
        self.test_loader = None
        self.in_features = in_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = MLPModel(in_features=in_features, num_classes=num_classes)
        self.best_model = self.model.state_dict()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = 'mps' if torch.backends.mps.is_available else 'cpu'
        self.device = 'cuda' if torch.cuda.is_available() else self.device
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def fit(self, x, y):
        """
        Fit the model on the given train_loader and test_loader for num_epochs epochs.
        You have to call set_test_dataloader before calling the fit function.
        Parameters
        ----------
        x: np.ndarray
            The training embeddings
        y: np.ndarray
            The training labels
        Returns
        -------
        self
        """
        dl = DataLoader(ReviewDataSet(x, y), batch_size=self.batch_size, shuffle=True)
        max_f1 = 0
        for epoch in tqdm(range(self.num_epochs)):
            # Train
            self.model.train()
            l = 0
            for x_batch, y_batch in dl:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                y_preds = self.model(x_batch)
                loss = self.criterion(y_preds, y_batch)
                loss.backward()
                self.optimizer.step()
                l += loss.item()
            l /= len(dl)
            # Evaluation
            test_loss, _, _, f1_macro = self._eval_epoch(self.test_dataloader, self.model)
            print(f"epoch {epoch + 1}/{self.num_epochs} | loss: {l:.4f} | test loss: {test_loss:.4f} | F1 score: {f1_macro:.4f}")


            if f1_macro > max_f1:
                max_f1 = f1_macro
                self.best_model = self.model.state_dict()
                torch.save(self.best_model,'deep.pt')

    def predict(self, x):
        """
        Predict the labels on the given test_loader
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        Returns
        -------
        predicted_labels: list
            The predicted labels
        """
        test_dataset = ReviewDataSet(x, np.zeros(len(x)))  # Dummy labels
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        preds, _, _ = self.predict_with_dl(test_loader, self.model)
        return preds

    def _eval_epoch(self, dataloader: torch.utils.data.DataLoader, model):
        """
        Evaluate the model on the given dataloader. used for validation and test
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
        Returns
        -------
        eval_loss: float
            The loss on the given dataloader
        predicted_labels: list
            The predicted labels
        true_labels: list
            The true labels
        f1_score_macro: float
            The f1 score on the given dataloader
        """
        preds, true_labels, loss = self.predict_with_dl(dataloader, model)
        f1_macro = f1_score(true_labels, preds, average='macro')
        return loss, preds, true_labels, f1_macro


    def predict_with_dl(self, dataloader: torch.utils.data.DataLoader, model):
        preds = []
        true_labels = []
        l = 0
        model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y_preds = model(x)
                l += self.criterion(y_preds, y).item()
                preds.extend(torch.argmax(y_preds, dim=1).cpu().numpy())
                true_labels.extend(y.cpu().numpy())
        return preds, true_labels, l / len(dataloader)

    def set_test_dataloader(self, X_test, y_test):
        """
        Set the test dataloader. This is used to evaluate the model on the test set while training
        Parameters
        ----------
        X_test: np.ndarray
            The test embeddings
        y_test: np.ndarray
            The test labels
        Returns
        -------
        self
            Returns self
        """
        test_dataset = ReviewDataSet(X_test, y_test)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def prediction_report(self, x, y):
        """
        Get the classification report on the given test set
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        y: np.ndarray
            The test labels
        Returns
        -------
        str
            The classification report
        """
        y_pred = self.predict(x)
        return classification_report(y, y_pred)

# F1 Accuracy : 79%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    rl = ReviewLoader('IMDB Dataset.csv')
    rl.load_data()
    rl.get_embeddings()

    X_train, X_test, y_train, y_test = rl.split_data(test_data_ratio=0.25)

    classifier = DeepModelClassifier(in_features=100, num_classes=2, batch_size=32)
    classifier.set_test_dataloader(X_test, y_test)

    classifier.fit(X_train, y_train)
    report = classifier.prediction_report(X_test, y_test)
    print(report)
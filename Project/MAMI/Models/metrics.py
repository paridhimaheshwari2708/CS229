import torch
import numpy as np
from typing import Union
from sklearn.metrics import f1_score, precision_score, recall_score


class Metric:
    def __init__(self):
        self.function = None
        self.average = None

    def __call__(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> float:
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if len(predictions.shape) == 1 or predictions.shape[1] == 1:
            self.average = 'macro'
        else:
            self.average = 'weighted'
        return self.function(predictions, targets)


class Accuracy(Metric):
    def __init__(self):
        super(Accuracy, self).__init__()
        self.function = self.accuracy

    def accuracy(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> float:
        assert len(predictions.shape) == len(targets.shape)
        return ((predictions > 0.5).astype(float) == targets).mean(axis=0).mean()


class Precision(Metric):
    def __init__(self):
        super(Precision, self).__init__()
        self.function = self.precision_

    def precision_(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> float:
        return precision_score(
            targets, (predictions > 0.5).astype(float), average=self.average
        )


class Recall(Metric):
    def __init__(self):
        super(Recall, self).__init__()
        self.function = self.recall_

    def recall_(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> float:
        return recall_score(targets, (predictions > 0.5).astype(float), average=self.average)


class F1Score(Metric):
    def __init__(self):
        super(F1Score, self).__init__()
        self.function = self.f1_score_

    def f1_score_(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> float:
        return f1_score(targets, (predictions > 0.5).astype(float), average=self.average)


if __name__ == "__main__":

    predictions = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    targets = torch.tensor([[0, 1], [1, 0]])

    Acc = Accuracy()
    print(Acc(predictions, targets))

    metric = Precision()
    print(metric(predictions, targets))

    metric = Recall()
    print(metric(predictions, targets))

    metric = F1Score()
    print(metric(predictions, targets))
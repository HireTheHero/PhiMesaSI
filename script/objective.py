import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import free_memory


# Cross-entropy
## Definition
def entropy_loss(input_tensor):
    lsm = nn.LogSoftmax()
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.mean()
    return entropy


## Loss function
def cross_entropy_loss(outputs, labels):
    return F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))


# Phi
## mutual information loss function
def mutual_information_loss(outputs_logits, outputs):
    return cross_entropy_loss(outputs_logits, outputs) - entropy_loss(outputs_logits)


## Phi
def bigphi_loss(model, config, inputs, outputs):
    outputs_logits = model(inputs).logits
    return -(
        mutual_information_loss(outputs_logits, outputs)
        - mib_loss(evaluate_bipartition, config, outputs_logits, outputs)
    )


# MIB
## bipartition evaluation
def evaluate_bipartition(trial, X, Y, loss_func, d):
    try:
        # Suggest a bipartition size d1
        d1 = trial.suggest_int("d1", 1, d - 1)

        # Randomly shuffle indices to create a bipartition
        indices = np.arange(d)
        np.random.shuffle(indices)

        indices1 = indices[:d1]
        indices2 = indices[d1:]

        X1, Y1 = X[:, indices1, :], Y[:, indices1]
        X2, Y2 = X[:, indices2, :], Y[:, indices2]

        # Calculate the metric for the bipartition
        metric1 = loss_func(X1, Y1)
        metric2 = loss_func(X2, Y2)

        # Combine the metrics
        metric = metric1 + metric2

        # Report the metric to Optuna for pruning
        trial.report(metric, step=0)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

        return metric
    finally:
        free_memory()


## MIB loss function
def mib_loss(objective, config, X, Y):
    opcon = config["optuna"]
    assert (
        X.shape[:1] == Y.shape[:1]
    ), f"First two dimension of X and Y must match: {(X.shape, Y.shape)}"
    d = X.shape[1]
    # Create a study with a pruner and optimize
    pruner = optuna.pruners.MedianPruner(**opcon["pruner"])
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(
        lambda trial: objective(trial, X, Y, mutual_information_loss, d),
        **opcon["optimizer"],
    )

    return study.best_trial.value

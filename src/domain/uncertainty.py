import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import logging
import torch
from typing import Any, Tuple, List
from src.domain.prompt import is_answerable
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier
import os
import json
from omegaconf import ListConfig, DictConfig


def naive_entropy(log_probs):
    """
        Compute
        
        `E[-log p(x)] ~= -1/N sum_i log p(x_i)`, i.e. the average token likelihood.
    """

    return -np.mean(log_probs)


def supervised_approach_grid_CV(
    train_embeddings, is_false, config, eval_embeddings=None, eval_is_false=None
):
    """Fit classifier (LogisticRegression or SVM) with optional PCA and plot results."""

    logging.info('Accuracy of model on Task: %f.', 1 - torch.tensor(is_false).mean())

    X_train_full = torch.cat(train_embeddings, dim=0).cpu().numpy()
    y_train_full = np.array(is_false)

    logging.info(
        "X_train_full shape: %s, y_train_full shape: %s", X_train_full.shape, y_train_full.shape)

    n_features_to_keep = config.n_features_to_keep
    if X_train_full.shape[1] > n_features_to_keep:
        X_train_full = X_train_full[:, :n_features_to_keep]
        logging.info("Selected first %d features, new X_train_full shape: %s", n_features_to_keep, X_train_full.shape)
    else:
        logging.info("Number of features <= %d, keeping all features", n_features_to_keep)
    
    base_dir = "images"  
    folder_name = f"{config.num_samples}_{n_features_to_keep}"  
    output_dir = os.path.join(base_dir, folder_name)

    os.makedirs(output_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),  
        ('pca', PCA()), 
        ('clf', LogisticRegression()) 
    ])

    param_grid = build_param_grid(config.param_grid)
    logging.info("Hyperparameters: %s", param_grid)

    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    logging.info("Best params: %s", grid.best_params_)
    best_model = grid.best_estimator_

    if eval_embeddings is not None:
        X_eval = torch.cat(eval_embeddings, dim=0).cpu().numpy()
        y_eval = np.array(eval_is_false)
    if X_eval.shape[1] > n_features_to_keep:
        X_eval = X_eval[:, :n_features_to_keep]
    else:
        X_eval, y_eval = X_test, y_test

    splits = {
        'train': (X_train, y_train),
        'test': (X_test, y_test),
        'eval': (X_eval, y_eval)
    }

    metrics = {}
    y_preds_proba = {}

    for name, (X, y) in splits.items():
        y_pred = best_model.predict(X)
        y_pred_proba = best_model.predict_proba(X)[:, 1]
        y_preds_proba[name] = y_pred_proba

        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_proba)
        metrics.update({f'acc_{name}': acc, f'auroc_{name}': auc})

        plt.figure(figsize=(6,4))
        plt.hist(y_pred_proba[y==0], bins=20, alpha=0.5, label='Class 0')
        plt.hist(y_pred_proba[y==1], bins=20, alpha=0.5, label='Class 1')
        plt.title(
            f'Predicted probability distribution ({name}) - n_features {n_features_to_keep} - num_samples {config.num_samples}', 
            fontsize=10
        )
        plt.xlabel('Predicted probability')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'prob_dist_{name}_{n_features_to_keep}_{config.num_samples}.png'))
        plt.close()

        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.title(f'ROC Curve ({name}) - n_features {n_features_to_keep} - num_samples {config.num_samples}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig(os.path.join(output_dir, f'roc_curve_{name}_{n_features_to_keep}_{config.num_samples}.png'))
        plt.close()
    
    model_name = best_model.named_steps['clf'].__class__.__name__
    best_hyperparams = grid.best_params_.copy()  
    best_hyperparams['model'] = model_name
    if 'clf' in best_hyperparams:
        best_hyperparams['clf'] = best_hyperparams['clf'].__class__.__name__
    best_hyperparams = make_json_serializable(best_hyperparams)

    hyperparam_file = os.path.join(output_dir, f"best_hyperparams_{model_name}.json")
    with open(hyperparam_file, "w") as f:
        json.dump(best_hyperparams, f, indent=4)

    plt.figure(figsize=(6, 4))
    for name, (X, y) in splits.items():
        y_pred_proba = y_preds_proba[name]
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        auc = roc_auc_score(y, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{model_name} ({name}) AUC={auc:.3f}')

    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.title('ROC Curves for all splits')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'roc_curve_all_{model_name}_{n_features_to_keep}_{config.num_samples}.png'))
    plt.close()

    logging.info('Metrics: %s', metrics)
    return y_preds_proba['eval'], metrics, best_model


def build_embeddings(
    generations: Any,
    split_name: str,
    results_dict: Any
) -> Tuple[List[Any], List[float]]:
    """
    Build embeddings, accuracy, and answerable lists for a dataset split
    and update results_dict with is_false and unanswerable measures.
    
    Returns:
        embeddings: list of embeddings
        is_true: list of accuracies
    """
    embeddings, is_true, answerable = [], [], []

    for id_question in generations:
        most_likely_answer = generations[id_question]['most_likely_answer']
        embeddings.append(most_likely_answer['embedding'])
        is_true.append(most_likely_answer['accuracy'])
        answerable.append(is_answerable(generations[id_question]))
    
    is_false = [1.0 - val for val in is_true]
    unanswerable = [1.0 - val for val in answerable]

    results_dict[f'{split_name}_is_false'] = is_false
    results_dict[f'{split_name}_unanswerable'] = unanswerable

    logging.info(
        'Proportion of %s that cannot be answered: %f',
        split_name,
        np.mean(unanswerable)
    )

    return embeddings, is_false


def cosine_kernel(X, Y):
    return cosine_similarity(X, Y)


class CosineSVM3(SVC):
    def __init__(self, C=1.0):
        super().__init__(C=C, kernel=cosine_kernel, probability=True)


def build_param_grid(yaml_grid):
    """
    
    """
    grid = []

    for g in yaml_grid:
        new_g = {}

        pca_list = []
        for p in g.get('pca', []):
            if isinstance(p, str) and p.startswith("PCA_"):
                var = float(p.split("_")[1])
                pca_list.append(PCA(n_components=var))
            elif p == 'passthrough':
                pca_list.append('passthrough')
        new_g['pca'] = pca_list

        clf_list = []
        for c in g.get('clf', []):
            if c == 'LogisticRegression':
                clf_list.append(LogisticRegression(max_iter=1000))
            elif c == 'SVC':
                clf_list.append(SVC(probability=True))
            elif c == 'CosineSVM':
                clf_list.append(CosineSVM3())  
            elif c == "MLPClassifier":
                clf_list.append(MLPClassifier())
        new_g['clf'] = clf_list

        for k, v in g.items():
            if k in ['pca', 'clf']:
                continue
            if k == 'clf__hidden_layer_sizes':
                new_values = []
                for val in v:
                    if isinstance(val, str):
                        try:
                            new_values.append(eval(val))  
                        except Exception as e:
                            raise ValueError(f"Impossible de parser hidden_layer_sizes: {val}") from e
                    else:
                        new_values.append(val)
                new_g[k] = new_values
            else:
                new_g[k] = v

        grid.append(new_g)

    return grid


def calculate_p_true(
        model, question, most_probable_answer, brainstormed_answers,
        few_shot_prompt, hint=False):
    """Calculate p_true uncertainty metric."""

    if few_shot_prompt:
        prompt = few_shot_prompt + '\n'
    else:
        prompt = ''

    prompt += 'Question: ' + question
    prompt += '\nBrainstormed Answers: '
    for answer in brainstormed_answers + [most_probable_answer]:
        prompt += answer.strip() + '\n'
    prompt += 'Possible answer: ' + most_probable_answer + '\n'
    if not hint:
        prompt += 'Is the possible answer:\n'
        prompt += 'A) True\n'
        prompt += 'B) False\n'
        prompt += 'The possible answer is:'
    else:
        prompt += 'Do the brainstormed answers match the possible answer? Respond with A if they do, if they do not respond with B. Answer:'

    log_prob = model.get_p_true(prompt)

    return log_prob


def make_json_serializable(obj):
    if isinstance(obj, DictConfig):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, ListConfig):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj

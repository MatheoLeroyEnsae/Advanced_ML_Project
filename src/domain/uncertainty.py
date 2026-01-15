import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import logging
import torch


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


def naive_entropy(log_probs):
    """
        Compute
        
        `E[-log p(x)] ~= -1/N sum_i log p(x_i)`, i.e. the average token likelihood.
    """

    return -np.sum(log_probs) / len(log_probs)


def supervised_approach(train_embeddings, is_false, eval_embeddings=None, eval_is_false=None):
    """Fit linear classifier to embeddings to predict model correctness."""

    logging.info('Accuracy of model on Task: %f.', 1 - torch.tensor(is_false).mean())  # pylint: disable=no-member

    # Convert the list of tensors to a 2D tensor.
    train_embeddings_tensor = torch.cat(train_embeddings, dim=0)  # pylint: disable=no-member
    # Convert the tensor to a numpy array.
    embeddings_array = train_embeddings_tensor.cpu().numpy()

    # Split the data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(  # pylint: disable=invalid-name
        embeddings_array, is_false, test_size=0.2, random_state=42)  # pylint: disable=invalid-name

    # Fit a logistic regression model.
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict deterministically and probabilistically and compute accuracy and auroc for all splits.
    X_eval = torch.cat(eval_embeddings, dim=0).cpu().numpy()  # pylint: disable=no-member,invalid-name
    y_eval = eval_is_false

    Xs = [X_train, X_test, X_eval]  # pylint: disable=invalid-name
    ys = [y_train, y_test, y_eval]  # pylint: disable=invalid-name
    suffixes = ['train_train', 'train_test', 'eval']

    metrics, y_preds_proba = {}, {}

    for suffix, X, y_true in zip(suffixes, Xs, ys):  # pylint: disable=invalid-name

        # If suffix is eval, we fit a new model on the entire training data set
        # rather than just a split of the training data set.
        if suffix == 'eval':
            model = LogisticRegression()
            model.fit(embeddings_array, is_false)
            convergence = {
                'n_iter': model.n_iter_[0],
                'converged': (model.n_iter_ < model.max_iter)[0]}

        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        y_preds_proba[suffix] = y_pred_proba
        acc_p_ik_train = accuracy_score(y_true, y_pred)
        auroc_p_ik_train = roc_auc_score(y_true, y_pred_proba[:, 1])
        split_metrics = {
            f'acc_p_ik_{suffix}': acc_p_ik_train,
            f'auroc_p_ik_{suffix}': auroc_p_ik_train}
        metrics.update(split_metrics)

    logging.info('Metrics for p_ik classifier: %s.', metrics)

    # Return model predictions on the eval set.
    return y_preds_proba['eval'][:, 1]


def supervised_approach_2(train_embeddings, is_false, eval_embeddings=None, eval_is_false=None):
    """
    Fit an SVM classifier with Platt scaling (probabilities) to embeddings.
    Hyperparameters are chosen via cross-validation (classic SVM grid search).
    
    Args:
        train_embeddings: list of torch tensors [n_samples, embedding_dim]
        is_false: list/array of labels (0/1)
        eval_embeddings: list of torch tensors for evaluation (optional)
        eval_is_false: labels for evaluation (optional)
    
    Returns:
        y_pred_proba_eval: np.array of predicted probabilities on eval set
    """
    
    logging.info('Baseline accuracy on train: %.4f', 1 - torch.tensor(is_false).float().mean())

    # Convert list of tensors to 2D numpy array
    X_train_full = torch.cat(train_embeddings, dim=0).cpu().numpy()
    y_train_full = np.array(is_false)

    # Optional: split training into train/test internally
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    # 1️⃣ Define SVM with probability=True for Platt scaling
    svm = SVC(probability=True, random_state=42)

    # 2️⃣ Hyperparameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']  # only relevant for rbf/poly
    }

    # 3️⃣ Cross-validation grid search
    grid = GridSearchCV(svm, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    logging.info("Best SVM params: %s", grid.best_params_)

    # 4️⃣ Evaluate on internal train/test split
    splits = {'train': (X_train, y_train), 'test': (X_test, y_test)}
    metrics = {}
    for name, (X, y_true) in splits.items():
        y_pred = best_model.predict(X)
        y_pred_proba = best_model.predict_proba(X)[:, 1]
        metrics[f'acc_{name}'] = accuracy_score(y_true, y_pred)
        metrics[f'auroc_{name}'] = roc_auc_score(y_true, y_pred_proba)
    logging.info("Metrics on internal splits: %s", metrics)

    # 5️⃣ Fit on full training data for evaluation predictions
    best_model.fit(X_train_full, y_train_full)

    if eval_embeddings is not None and eval_is_false is not None:
        X_eval = torch.cat(eval_embeddings, dim=0).cpu().numpy()
        y_eval = np.array(eval_is_false)
        y_pred_proba_eval = best_model.predict_proba(X_eval)[:, 1]

        acc_eval = accuracy_score(y_eval, best_model.predict(X_eval))
        auroc_eval = roc_auc_score(y_eval, y_pred_proba_eval)
        logging.info("Eval metrics: accuracy=%.4f, auroc=%.4f", acc_eval, auroc_eval)
    else:
        y_pred_proba_eval = None

    return y_pred_proba_eval
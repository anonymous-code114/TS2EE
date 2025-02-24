import torch
import sys
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
torch.manual_seed(3407)
sys.path.append('../')
from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, classification_report
from get_features import get_features_from_encoder


def Eval(train_loader, test_loader, model_checkpoints_folder, encoder, task, fold):
    # remove the projection head
    encoder.eval()
    x_train, y_train = get_features_from_encoder(encoder, train_loader)
    x_test, y_test = get_features_from_encoder(encoder, test_loader)
    x_train = x_train.cpu().numpy()
    y_train = y_train.cpu().numpy()
    x_test = x_test.cpu().numpy()
    y_test = y_test.cpu().numpy()
    x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
    x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train_reshaped)
    torch.cuda.empty_cache()
    print("Linear model evaluation")
    # acc=linear_model_eval(scaler.transform(x_train_reshaped), y_train, scaler.transform(x_test_reshaped), y_test,
    #                   model_checkpoints_folder, task, fold)

    acc = svm_model_eval(scaler.transform(x_train_reshaped), y_train, scaler.transform(x_test_reshaped), y_test,
                         model_checkpoints_folder, task, fold)
    print(acc)
    return acc


def linear_model_eval(X_train, y_train, X_test, y_test, model_checkpoints_folder, task, fold):
    """
    Evaluates the model using logistic regression and calculates various metrics.
    """
    clf = LogisticRegression(random_state=0, max_iter=10000, solver='lbfgs', C=1.0)
    print(X_train, y_train)
    clf.fit(X_train, y_train)

    # Predictions
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # Open a file to write the metrics
    with open(os.path.join(model_checkpoints_folder, 'checkpoints',
                           'evaluation_results_' + str(fold) + "_" + task + '_linear.txt'),
              'w') as file:
        file.write("Logistic Regression feature eval\n")
        file.write("Train Metrics:\n")
        file.write(f"Accuracy: {accuracy_score(y_train, y_train_pred)}\n")
        file.write("Classification Report:\n")
        file.write(f"{classification_report(y_train, y_train_pred)}\n")
        file.write(f"Precision: {precision_score(y_train, y_train_pred, average='macro')}\n")
        file.write(f"Recall: {recall_score(y_train, y_train_pred, average='macro')}\n")
        file.write(f"F1 Score: {f1_score(y_train, y_train_pred, average='macro')}\n")

        file.write("\nTest Metrics:\n")
        file.write(f"Accuracy: {accuracy_score(y_test, y_test_pred)}\n")
        file.write("Classification Report:\n")
        file.write(f"{classification_report(y_test, y_test_pred)}\n")
        file.write(f"Precision: {precision_score(y_test, y_test_pred, average='macro')}\n")
        file.write(f"Recall: {recall_score(y_test, y_test_pred, average='macro')}\n")
        file.write(f"F1 Score: {f1_score(y_test, y_test_pred, average='macro')}\n")

        # Assuming binary classification for AUROC and AUPRC
        if len(np.unique(y_test)) == 2:
            y_test_prob = clf.predict_proba(X_test)[:, 1]
            file.write(f"AUROC: {roc_auc_score(y_test, y_test_prob)}\n")
            file.write(f"AUPRC: {average_precision_score(y_test, y_test_prob)}\n")
        else:
            file.write("AUROC and AUPRC are not calculated because the problem is not binary classification.\n")

        file.write("-------------------------------\n")
        file.close()
    acc = accuracy_score(y_test, y_test_pred)
    return acc


def svm_model_eval(X_train, y_train, X_test, y_test, model_checkpoints_folder, task, fold):
    """
    Evaluates the model using SVM and calculates various metrics.
    """
    # Predictions
    clf = svm.SVC(kernel='rbf', C=1.0, probability=True)
    #print(X_train, y_train)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # Open a file to write the metrics
    with open(os.path.join(model_checkpoints_folder, 'checkpoints',
                           'evaluation_results_' + str(fold) + "_" + task + '_linear.txt'),
              'w') as file:
        file.write("Logistic Regression feature eval\n")
        file.write("Train Metrics:\n")
        file.write(f"Accuracy: {accuracy_score(y_train, y_train_pred)}\n")
        file.write("Classification Report:\n")
        file.write(f"{classification_report(y_train, y_train_pred)}\n")
        file.write(f"Precision: {precision_score(y_train, y_train_pred, average='macro')}\n")
        file.write(f"Recall: {recall_score(y_train, y_train_pred, average='macro')}\n")
        file.write(f"F1 Score: {f1_score(y_train, y_train_pred, average='macro')}\n")

        file.write("\nTest Metrics:\n")
        file.write(f"Accuracy: {accuracy_score(y_test, y_test_pred)}\n")
        file.write("Classification Report:\n")
        file.write(f"{classification_report(y_test, y_test_pred)}\n")
        file.write(f"Precision: {precision_score(y_test, y_test_pred, average='macro')}\n")
        file.write(f"Recall: {recall_score(y_test, y_test_pred, average='macro')}\n")
        file.write(f"F1 Score: {f1_score(y_test, y_test_pred, average='macro')}\n")

        # Assuming binary classification for AUROC and AUPRC
        if len(np.unique(y_test)) == 2:
            y_test_prob = clf.predict_proba(X_test)[:, 1]
            file.write(f"AUROC: {roc_auc_score(y_test, y_test_prob)}\n")
            file.write(f"AUPRC: {average_precision_score(y_test, y_test_prob)}\n")
        else:
            file.write("AUROC and AUPRC are not calculated because the problem is not binary classification.\n")

        file.write("-------------------------------\n")
    acc = accuracy_score(y_test, y_test_pred)
    return acc

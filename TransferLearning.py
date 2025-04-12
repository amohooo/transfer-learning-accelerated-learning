# -*- coding: utf-8 -*-
'''

The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.

No partial marks will be awarded for functions that do not meet the specifications
of the interfaces.

Last modified 2024-05-07 by Anthony Vanderkop.
Hopefully without introducing new bugs.
'''


### LIBRARY IMPORTS HERE ###
import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical, load_img, img_to_array
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
import seaborn as sns
import matplotlib.pyplot as plt
    
def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [(11427591, 'Mohan', 'Hao'), (11308826, 'Olivia', 'Zheng')]
    
def load_model():
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
    base_model.trainable = False  # Freeze the layers
    return base_model

def load_data(path):
    labels = []
    images = []
    class_names = [] 
    classes = {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    for label, encoding in classes.items():
        class_dir = os.path.join(path, label)
        class_names.append(label)
        for file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, file)
            img = load_img(img_path, target_size=(224, 224))
            images.append(img_to_array(img))
            labels.append(encoding)
    images = (np.array(images, dtype='float32') / 127.5) - 1  # Normalize to [-1, 1]
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=5)  # One-hot encode labels
    return images, labels, class_names

def split_data(X, Y, train_fraction, randomize=False, eval_set=True):
    # Convert one-hot encoded labels back to single labels for stratification
    labels = np.argmax(Y, axis=1)
    unique_classes = np.unique(labels)
    class_indices = {cls: np.where(labels == cls)[0] for cls in unique_classes}
    train_indices = []
    test_indices = []
    eval_indices = []
    for cls, indices in class_indices.items():
        if randomize:
            np.random.shuffle(indices)
        num_train = int(len(indices) * train_fraction)
        num_test = len(indices) - num_train
        num_eval = 0
        if eval_set:
            num_eval = num_test // 2
            num_test -= num_eval
        train_indices.extend(indices[:num_train])
        test_indices.extend(indices[num_train:num_train + num_test])
        if eval_set:
            eval_indices.extend(indices[num_train + num_test:])
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]
    if eval_set:
        X_eval, Y_eval = X[eval_indices], Y[eval_indices]
        return (X_train, Y_train), (X_test, Y_test), (X_eval, Y_eval)
    else:
        return (X_train, Y_train), (X_test, Y_test)

def confusion_matrix(test_labels, predictions, class_names, plot=False):
    if test_labels.ndim > 1:
        test_labels = np.argmax(test_labels, axis=1)
    cm = sk_confusion_matrix(test_labels, predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if plot:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.show()
    return cm

def precision(predictions, true_labels, class_names):
    precision_scores = np.zeros(len(class_names))
    for i, class_name in enumerate(class_names):
        tp = np.sum((predictions == i) & (true_labels == i))
        fp = np.sum((predictions == i) & (true_labels != i))
        precision_scores[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
    return precision_scores

def recall(predictions, true_labels, class_names):
    recall_scores = np.zeros(len(class_names))
    for i, class_name in enumerate(class_names):
        tp = np.sum((predictions == i) & (true_labels == i))
        fn = np.sum((predictions != i) & (true_labels == i))
        recall_scores[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
    return recall_scores

def f1(predictions, true_labels, class_names):
    prec = precision(predictions, true_labels, class_names)
    rec = recall(predictions, true_labels, class_names)
    f1 = 2 * (prec * rec) / (prec + rec)
    f1 = np.nan_to_num(f1)  # Handles division by zero in case of zero precision and recall
    return f1

def k_fold_validation(features, ground_truth, classifier_func, class_names, k=5):
    n = len(features)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k
    all_metrics = []
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size if i < k - 1 else n]
        train_indices = np.setdiff1d(indices, test_indices)
        train_features = features[train_indices]
        train_labels = ground_truth[train_indices]
        test_features = features[test_indices]
        test_labels = ground_truth[test_indices]
        metrics = classifier_func(train_features, train_labels, test_features, test_labels)
        all_metrics.append(metrics)
    all_metrics = np.array(all_metrics)  # Shape should be [k, 3] if metrics returned as [precision, recall, f1]
    avg_metrics = np.mean(all_metrics, axis=0)
    sigma_metrics = np.std(all_metrics, axis=0)
    return avg_metrics, sigma_metrics

def transfer_learning(train_set, eval_set, test_set, base_model, parameters, class_names):
    learning_rate, momentum, nesterov = parameters
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(class_names), activation='softmax')(x)
    new_model = Model(inputs=base_model.input, outputs=predictions)
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
    new_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    new_model.fit(train_set[0], train_set[1], validation_data=eval_set, epochs=10)
    loss, accuracy = new_model.evaluate(test_set[0], test_set[1])
    predictions = new_model.predict(test_set[0])
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(test_set[1], axis=1)
    precision_scores = precision(predicted_classes, actual_classes, class_names)
    recall_scores = recall(predicted_classes, actual_classes, class_names)
    f1_scores = f1(predicted_classes, actual_classes, class_names)
    metrics = [precision_scores, recall_scores, f1_scores]
    new_model.save("trained_model_transfer.h5")
    return new_model, metrics

def accelerated_learning(train_set, eval_set, test_set, base_model, parameters, class_names):
    train_features = base_model.predict(train_set[0])
    if eval_set is not None:
        eval_features = base_model.predict(eval_set[0])
    else:
        eval_features = None
    if test_set is not None:
        test_features = base_model.predict(test_set[0])
        test_labels = test_set[1]
    else:
        return None, "No test set provided"
    top_model = Sequential([
        GlobalAveragePooling2D(input_shape=train_features.shape[1:]),
        Dense(1024, activation='relu'),
        Dense(len(class_names), activation='softmax')
    ])
    learning_rate, momentum, nesterov = parameters
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
    top_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    if eval_features is not None:
        top_model.fit(train_features, train_set[1], validation_data=(eval_features, eval_set[1]), epochs=10)
    else:
        top_model.fit(train_features, train_set[1], epochs=10)
    loss, accuracy = top_model.evaluate(test_features, test_labels)
    predictions = top_model.predict(test_features)
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(test_labels, axis=1)
    precision_scores = precision(predicted_classes, actual_classes, class_names)
    recall_scores = recall(predicted_classes, actual_classes, class_names)
    f1_scores = f1(predicted_classes, actual_classes, class_names)
    metrics = [precision_scores, recall_scores, f1_scores]
    top_model.save("trained_model_accelerated.h5")
    return top_model, metrics

def transfer_learning_classifier(train_features, train_labels, test_features, test_labels, base_model, parameters, class_names):
    _, metrics = transfer_learning((train_features, train_labels), None, (test_features, test_labels), base_model, parameters, class_names)
    return metrics

def accelerated_learning_classifier(train_features, train_labels, test_features, test_labels, base_model, parameters, class_names):
    _, metrics = accelerated_learning((train_features, train_labels), None, (test_features, test_labels), base_model, parameters, class_names)
    return metrics

if __name__ == "__main__":
    data_path = '/home/imhaom/CAB420_Machine_Learning/CAB320/small_flower_dataset'
    model = load_model()
    images, labels, class_names = load_data(data_path)
    (X_train, Y_train), (X_test, Y_test), (X_eval, Y_eval) = split_data(images, labels, train_fraction=0.8, randomize=True, eval_set=True)
    parameters = (0.001, 0.9, True)
    def print_class_distribution(labels, label_names):
        class_counts = np.sum(labels, axis=0)
        for name, count in zip(label_names, class_counts):
            print(f"{name}: {count}")
            
    # After loading and splitting the data
    print("Training class distribution:")
    print_class_distribution(Y_train, class_names)

    print("Testing class distribution:")
    print_class_distribution(Y_test, class_names)

    print("Validation class distribution:")
    print_class_distribution(Y_eval, class_names)
    # Transfer Learning
    trained_model_transfer, metrics_transfer = transfer_learning((X_train, Y_train), (X_eval, Y_eval), (X_test, Y_test), model, parameters, class_names)
    # trained_model_transfer.save("trained_model_transfer.h5")
    trained_model_transfer.summary()
    predictions_transfer = trained_model_transfer.predict(X_test)
    predicted_classes_transfer = np.argmax(predictions_transfer, axis=1)
    actual_classes_transfer = np.argmax(Y_test, axis=1)
    print("Transfer Learning Model Performance:", metrics_transfer)
    confusion_matrix(actual_classes_transfer, predicted_classes_transfer, class_names, plot=True)
    
    # Accelerated Learning
    trained_model_accelerated, metrics_accelerated = accelerated_learning((X_train, Y_train), (X_eval, Y_eval), (X_test, Y_test), model, parameters, class_names)
    # trained_model_accelerated.save("trained_model_accelerated.h5")
    trained_model_accelerated.summary()
    test_features_accelerated = model.predict(X_test)
    predictions_accelerated = trained_model_accelerated.predict(test_features_accelerated)
    predicted_classes_accelerated = np.argmax(predictions_accelerated, axis=1)
    actual_classes_accelerated = np.argmax(Y_test, axis=1)
    print("Accelerated Learning Model Performance:", metrics_accelerated)
    confusion_matrix(actual_classes_accelerated, predicted_classes_accelerated, class_names, plot=True)
    
    # K-Fold Validation for Transfer Learning
    avg_metrics_transfer, sigma_metrics_transfer = k_fold_validation(
        images, labels,
        lambda X_train, Y_train, X_test, Y_test: transfer_learning_classifier(
            X_train, Y_train, X_test, Y_test, model, parameters, class_names
        ),
        class_names, k=5
    )
    print("Transfer Learning - Avg Metrics:\n", avg_metrics_transfer)
    print("Transfer Learning - Sigma Metrics:\n", sigma_metrics_transfer)
    
    # K-Fold Validation for Accelerated Learning
    avg_metrics_accelerated, sigma_metrics_accelerated = k_fold_validation(
        images, labels,
        lambda X_train, Y_train, X_test, Y_test: accelerated_learning_classifier(
            X_train, Y_train, X_test, Y_test, model, parameters, class_names
        ),
        class_names, k=5
    )
    print("Accelerated Learning - Avg Metrics:\n", avg_metrics_accelerated)
    print("Accelerated Learning - Sigma Metrics:\n", sigma_metrics_accelerated)

    
#########################  CODE GRAVEYARD  #############################

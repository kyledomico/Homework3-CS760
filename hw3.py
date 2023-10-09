# Write a function that will plot a ROC curve with given data with positive confidence scores and labels.
import matplotlib.pyplot as plt
import numpy as np

data = [[0.95,1],[0.85,1],[0.8,0],[0.7,1], [0.55,1],[0.45,0],[0.4,1],[0.3,1],[0.2,0],[0.1,0]]

def plot_roc(data):
    # Sort the data by confidence score
    data = sorted(data, key=lambda x: x[0], reverse=True)
    # Get the total number of positive and negative examples
    total_pos = sum([1 for x in data if x[1] == 1])
    total_neg = sum([1 for x in data if x[1] == 0])
    # Initialize the true positive and false positive counts
    tp = 0
    fp = 0
    # Initialize the true positive rate and false positive rate lists
    tpr_list = []
    fpr_list = []
    # Loop through the data
    for i in range(len(data)):
        # If the label is positive, increment the true positive count
        if data[i][1] == 1:
            tp += 1
        # If the label is negative, increment the false positive count
        else:
            fp += 1
        # Compute the true positive rate and false positive rate
        tpr = tp / total_pos
        fpr = fp / total_neg
        # Append the true positive rate and false positive rate to their lists
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    print(list(zip(fpr_list, tpr_list)))
    # Plot the ROC curve
    plt.plot(fpr_list, tpr_list)
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig('roc_curve.pdf')
    plt.clf()

# 1. Use the whole D2z.txt as training set. Use Euclidean distance (i.e. A = I). Visualize the predictions
# of 1NN on a 2D grid [−2 : 0.1 : 2]2. That is, you should produce test points whose first feature goes over
# −2, −1.9, −1.8, . . . , 1.9, 2, so does the second feature independent of the first feature. You should overlay
# the training set in the plot, just make sure we can tell which points are training, which are grid.
# Now, we will use ’emails.csv’ as our dataset. The description is as follows.
# • Task: spam detection
# • The number of rows: 5000
# • The number of features: 3000 (Word frequency in each email)
# • The label (y) column name: ‘Predictor’
# • For a single training/test set split, use Email 1-4000 as the training set, Email 4001-5000 as the test
# set.
# • For 5-fold cross validation, split dataset in the following way.
# – Fold 1, test set: Email 1-1000, training set: the rest (Email 1001-5000)
# – Fold 2, test set: Email 1000-2000, training set: the rest
# – Fold 3, test set: Email 2000-3000, training set: the rest
# – Fold 4, test set: Email 3000-4000, training set: the rest
# – Fold 5, test set: Email 4000-5000, training set: the rest
import pandas as pd

# Read the D2z.txt file
df = pd.read_csv('hw3Data/D2z.txt', sep=' ', header=None).to_numpy()

# Define 1NN function
def one_nn(x, y, z):
    # Initialize the minimum distance and the minimum distance index
    min_dist = float('inf')
    min_dist_index = None
    # Loop through the training set
    for i in range(len(x)):
        # Compute the distance between the test point and the training point
        dist = np.linalg.norm(z - x[i])
        # If the distance is less than the minimum distance, update the minimum distance and the minimum distance index
        if dist < min_dist:
            min_dist = dist
            min_dist_index = i
    # Return the label of the nearest neighbor
    return y[min_dist_index]

# Define the test set grid
x1 = [i/10 for i in range(-20, 21)]
x2 = [i/10 for i in range(-20, 21)]
test = np.array([[i, j] for i in x1 for j in x2])

# Run 1NN on the test set and plot the classifications
predictions = [one_nn(df[:,0:2], df[:,2], test[i]) for i in range(len(test))]
plt.scatter(test[:,0], test[:,1], c=predictions, cmap='viridis')
plt.scatter(df[:,0], df[:,1], c=df[:,2])
plt.savefig('1nn.pdf')
plt.clf()

# Read the emails.csv file
df = pd.read_csv('hw3Data/emails.csv')

# Define function returning single run training set (Emails 1-4000) and test set (Emails 4001-5000) in Numpy arrays
def single_run(data):
    train_features = data.iloc[0:4000, 1:-1].to_numpy()
    train_labels = data.iloc[0:4000, -1].to_numpy()
    test_features = data.iloc[4000:5000, 1:-1].to_numpy()
    test_labels = data.iloc[4000:5000, -1].to_numpy()
    return train_features, train_labels, test_features, test_labels

# Define function returning one fold of the 5-fold cross validation training and test sets in Numpy arrays
def five_fold_select(data, fold):
    # Test Set Indices
    test_start = (fold - 1) * 1000
    test_end = fold * 1000
    
    # Test Set
    test_features = data.iloc[test_start:test_end, 1:-1].to_numpy()
    test_labels = data.iloc[test_start:test_end, -1].to_numpy()

    # Training Dataframe
    train_df = data.drop(data.index[test_start:test_end])
    train_features = train_df.iloc[:, 1:-1].to_numpy()
    train_labels = train_df.iloc[:, -1].to_numpy()

    return train_features, train_labels, test_features, test_labels

# 2. Implement 1NN, Run 5-fold cross validation. Report accuracy, precision, and recall in each fold.
# Run one_nn on each fold of the 5-fold cross validation
def run_1nn_5fold(df):
    for i in range(1, 6):
        train_features, train_labels, test_features, test_labels = five_fold_select(df, i)
        predictions = [one_nn(train_features, train_labels, test_features[j]) for j in range(len(test_features))]
        # Compute the accuracy, precision, and recall
        accuracy = sum([1 for j in range(len(test_labels)) if predictions[j] == test_labels[j]]) / len(test_labels)
        precision = sum([1 for j in range(len(test_labels)) if predictions[j] == test_labels[j] == 1]) / sum([1 for j in range(len(test_labels)) if predictions[j] == 1])
        recall = sum([1 for j in range(len(test_labels)) if predictions[j] == test_labels[j] == 1]) / sum([1 for j in range(len(test_labels)) if test_labels[j] == 1])
        print('Fold', i, 'Accuracy:', accuracy, 'Precision:', precision, 'Recall:', recall)

# run_1nn_5fold(df)

# 3. Implement logistic regression (from scratch). Use gradient descent (refer to question 6 from part 1)
# to find the optimal parameters. You may need to tune your learning rate to find a good optimum. Run 5-fold
# cross validation. Report accuracy, precision, and recall in each fold.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_regression(x, y, alpha, iterations):
    # Initialize the weights
    w = np.zeros(x.shape[1])
    # Loop through the iterations
    for i in range(iterations):
        # Compute the gradient
        gradient = np.dot(x.T, (sigmoid(np.dot(x, w)) - y)) / y.size
        # Update the weights
        w -= alpha * gradient
    # Return the weights
    return w

def run_logistic_regression_5fold(df, alpha, iterations):
    for i in range(1, 6):
        train_features, train_labels, test_features, test_labels = five_fold_select(df, i)
        # Add bias term
        train_features = np.c_[np.ones(train_features.shape[0]), train_features]
        test_features = np.c_[np.ones(test_features.shape[0]), test_features]
        # Train the model
        w = logistic_regression(train_features, train_labels, alpha, iterations)
        # Make predictions
        predictions = [1 if sigmoid(np.dot(w, test_features[j])) > 0.5 else 0 for j in range(len(test_features))]
        # Compute the accuracy, precision, and recall
        accuracy = sum([1 for j in range(len(test_labels)) if predictions[j] == test_labels[j]]) / len(test_labels)
        precision = sum([1 for j in range(len(test_labels)) if predictions[j] == test_labels[j] == 1]) / sum([1 for j in range(len(test_labels)) if predictions[j] == 1])
        recall = sum([1 for j in range(len(test_labels)) if predictions[j] == test_labels[j] == 1]) / sum([1 for j in range(len(test_labels)) if test_labels[j] == 1])
        print('Fold', i, 'Accuracy:', accuracy, 'Precision:', precision, 'Recall:', recall)

# run_logistic_regression_5fold(df, 0.01, 1000)

# 4. Run 5-fold cross validation with kNN varying k (k=1, 3, 5, 7, 10). Plot the average accuracy versus
# k, and list the average accuracy of each case.
def kNN(x, y, z, k):
    # Initialize the list of distances
    distances = []
    # Loop through the training set
    for i in range(len(x)):
        # Compute the distance between the test point and the training point
        dist = np.linalg.norm(z - x[i])
        # Append the distance and the label to the list of distances
        distances.append([dist, y[i]])
    # Sort the list of distances by distance
    distances = sorted(distances, key=lambda x: x[0])
    # Get the k nearest neighbors
    neighbors = distances[:k]
    # Return the label of the majority of the k nearest neighbors
    return max(set([x[1] for x in neighbors]), key=[x[1] for x in neighbors].count)

def run_kNN_5fold(df, k):
    average_accuracy = 0
    for i in range(1, 6):
        train_features, train_labels, test_features, test_labels = five_fold_select(df, i)
        # Make predictions
        predictions = [kNN(train_features, train_labels, test_features[j], k) for j in range(len(test_features))]
        # Compute the accuracy, precision, and recall
        accuracy = sum([1 for j in range(len(test_labels)) if predictions[j] == test_labels[j]]) / len(test_labels)
        print('Fold', i, 'Accuracy:', accuracy)
        average_accuracy += accuracy
    average_accuracy = average_accuracy / 5
    return average_accuracy

# k_values = [1, 3, 5, 7, 10]
# average_accuracies = [run_kNN_5fold(df, k) for k in k_values]
# plt.plot(k_values, average_accuracies, c='r')
# plt.xlabel('k')
# plt.ylabel('Average Accuracy')
# plt.title('kNN Accuracy vs. k')
# plt.savefig('knn_accuracy.pdf')
# plt.clf()

# 5. Use a single training/test setting. Train kNN (k=5) and logistic regression on the training set, and
# draw ROC curves based on the test set.

# Get the Train/Test Sets
train_features, train_labels, test_features, test_labels = single_run(df)

# Use sklearn to train kNN and logistic regression
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Train kNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_features, train_labels)

# Train logistic regression
logistic = LogisticRegression()
logistic.fit(train_features, train_labels)

# Get the kNN and logistic regression predictions and plot the ROC curves
knn_predictions = knn.predict_proba(test_features)[:,1]
logistic_predictions = logistic.predict_proba(test_features)[:,1]

# Plot the ROC Curves with sklearn
from sklearn import metrics
knn_fpr, knn_tpr, knn_thresholds = metrics.roc_curve(test_labels, knn_predictions)
logistic_fpr, logistic_tpr, logistic_thresholds = metrics.roc_curve(test_labels, logistic_predictions)

plt.plot(knn_fpr, knn_tpr, label='kNN', c='b')
plt.plot(logistic_fpr, logistic_tpr, label='Logistic Regression', c='g')
plt.legend()
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.savefig('knn_logistic_roc.pdf')
plt.clf()


# # Implement knn that returns zipped confidences and predictions
# def knn_with_confidence(x, y, z, k):
#     # Initialize the list of distances
#     distances = []
#     # Loop through the training set
#     for i in range(len(x)):
#         # Compute the distance between the test point and the training point
#         dist = np.linalg.norm(z - x[i])
#         # Append the distance and the label to the list of distances
#         distances.append([dist, y[i]])
#     # Sort the list of distances by distance
#     distances = sorted(distances, key=lambda x: x[0])
#     # Get the k nearest neighbors
#     neighbors = distances[:k]
#     # Return the label of the majority of the k nearest neighbors
#     prediction = max(set([x[1] for x in neighbors]), key=[x[1] for x in neighbors].count), [x[1] for x in neighbors]
#     confidence = [x[1] for x in neighbors].count(prediction[0]) / k
#     return confidence, prediction[0]

# # Get single training/test set
# train_features, train_labels, test_features, test_labels = single_run(df)

# # Get Confidences and Predictions on the Test Set for kNN
# knn_confidences = [knn_with_confidence(train_features, train_labels, test_features[i], 5) for i in range(len(test_features))]

# # Get Confidences and Predictions on the Test Set for Logistic Regression
# # Add bias term
# train_features = np.c_[np.ones(train_features.shape[0]), train_features]
# test_features = np.c_[np.ones(test_features.shape[0]), test_features]

# # Train the model
# w = logistic_regression(train_features, train_labels, 0.01, 1000)

# # Zip test confidences and predictions
# logistic_confidences = [sigmoid(np.dot(w, test_features[i])) for i in range(len(test_features))]
# logistic_predictions = [1 if logistic_confidences[i] >= 0.5 else 0 for i in range(len(test_features))]

# # Write a Function that Plots ROC Curves
# def get_roc(data):
#     # Sort the data by confidence score
#     data = sorted(data, key=lambda x: x[0], reverse=True)
#     # Get the total number of positive and negative examples
#     total_pos = sum([1 for x in data if x[1] == 1])
#     total_neg = sum([1 for x in data if x[1] == 0])
#     # Initialize the true positive and false positive counts
#     tp = 0
#     fp = 0
#     # Initialize the true positive rate and false positive rate lists
#     tpr_list = []
#     fpr_list = []
#     # Loop through the data
#     for i in range(len(data)):
#         # If the label is positive, increment the true positive count
#         if data[i][1] == 1:
#             tp += 1
#         # If the label is negative, increment the false positive count
#         else:
#             fp += 1
#         # Compute the true positive rate and false positive rate
#         tpr = tp / total_pos
#         fpr = fp / total_neg
#         # Append the true positive rate and false positive rate to their lists
#         tpr_list.append(tpr)
#         fpr_list.append(fpr)
    
#     return list(zip(fpr_list, tpr_list))

# # Get ROC Curves for kNN and Logistic Regression
# knn_roc = get_roc(knn_confidences)
# logistic_roc = get_roc(list(zip(logistic_confidences, logistic_predictions)))

# # Plot the ROC Curves with sklearn
# import sklearn
# from sklearn import metrics

# # Get the fpr and tpr for kNN
# knn_fpr, knn_tpr, knn_thresholds = metrics.roc_curve(test_labels, [x[0] for x in knn_confidences])
# # Get the fpr and tpr for Logistic Regression
# logistic_fpr, logistic_tpr, logistic_thresholds = metrics.roc_curve(test_labels, logistic_confidences)

# # Plot the ROC Curves
# plt.plot(knn_fpr, knn_tpr, label='kNN', c='b')
# plt.plot(logistic_fpr, logistic_tpr, label='Logistic Regression', c='g')
# plt.legend()
# plt.xlim(-0.1,1.1)
# plt.ylim(-0.1,1.1)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curves')
# plt.savefig('knn_logistic_roc.pdf')
# plt.clf()

# # plt.plot([x[0] for x in knn_roc], [x[1] for x in knn_roc], label='kNN', c='b')
# # plt.plot([x[0] for x in logistic_roc], [x[1] for x in logistic_roc], label='Logistic Regression', c='g')
# # plt.legend()
# # plt.xlim(-0.1,1.1)
# # plt.ylim(-0.1,1.1)
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('ROC Curves')
# # plt.savefig('knn_logistic_roc.pdf')
# # plt.clf()




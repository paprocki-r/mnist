import matplotlib.pyplot as plt
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from get_dataset import get_dataset

# Turn down for faster convergence
train_samples = 5000

X_train, X_test, y_train, y_test = get_dataset(train_samples)
scaler = StandardScaler()
X_train = scaler.fit_transform(y_train) # 2. Preprocessing bug: y_train -> X_train
X_test = scaler.transform(X_test)

# Turn up tolerance for faster convergence
clf = LogisticRegression(C=50.0 / train_samples, penalty="l1", solver="sag", tol=0.1) # 3. Model Definition Bug: either "sag" -> "saga" or  penalty = "l1" -> penalty = "l2" - why?
clf.fit(X_test, y_test) # 4. Model Training Bug:  X_test -> X_train ,  y_test -> y_train
sparsity = np.mean(clf.coef_ == 0) * 100
score = clf.score(X_train, y_train) # 5. Evaluation bug: X_train -> X_test,   y_train -> y_test  - what is score? (Answer: accuracy_score)
# print('Best C % .4f' % clf.C_)
print("Sparsity with L1 penalty: %.2f%%" % sparsity)
print("Test score with L1 penalty: %.4f" % score)

plt.figure(figsize=(10, 5))
for i in range(10):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(
        X_test[i].reshape(20, 20), #  2. Preprocessing bug: (20, 20) -> (28, 28)
        interpolation="nearest",
    )
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel("Class %s" % y_test[i])
plt.suptitle("Classification vector for...")

plt.savefig("mnist-results.png")

import numpy as np
import matplotlib.pyplot as plt
import utils
import time

from support_vector_machines import SVM


# Helper functions to draw decision boundary plot
def plot_contours(clf, X, y, n=100):
    """
    Produce classification decision boundary

    Args:
        clf:
            Any classifier object that predicts {-1, +1} labels
        
        X (numpy.array):
            A 2d feature matrix

        y (numpy.array):
            A {-1, +1} label vector

        n (int)
            Number of points to partition the meshgrids
            Default = 100.

    Returns:
        (fig, ax)
            fig is the figure handle
            ax is the single axis in the figure

        One can use fig to save the figure.
        Or ax to modify the title/axis label etc

    """
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    X0, X1 = X[:, 0], X[:, 1]

    # Set-up grid for plotting.
    xx, yy = np.meshgrid(np.linspace(X0.min()-1, X0.max()+1, n),\
                         np.linspace(X1.min()-1, X1.max()+1, n),\
                        )
    # Do prediction for every single point on the mesh grid
    # This will take a few seconds
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=ListedColormap(["cyan", "pink"]))

    # Scatter the -1 points
    ax.scatter([X0[i] for i,v in enumerate(y) if v == -1],
                        [X1[i] for i,v in enumerate(y) if v == -1], 
                        c="blue", label='- 1',
                        marker='x')
    # Scatter the +1 points
    ax.scatter([X0[i] for i,v in enumerate(y) if v == 1],
                        [X1[i] for i,v in enumerate(y) if v == 1], 
                        edgecolor="red", label='+1', facecolors='none', s=10,
                        marker='o')

    ax.set_ylabel('x_2')
    ax.set_xlabel('x_1')
    ax.legend()
    return fig, ax


# Your code starts here.

# First load the data
X_syn_train, y_syn_train, X_syn_test, y_syn_test = utils.load_all_train_test_data("/Users/ceciliz/Desktop/ps2_kit/P1/Synthetic-Dataset")
synthetic_folds = utils.load_all_cross_validation_data("/Users/ceciliz/Desktop/ps2_kit/P1/Synthetic-Dataset/CrossValidation")


C_list = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 1, 10, 10**(2)]

d1_train_error = []
d1_test_error = []

for i in C_list:
    lin_kernel = lambda x,y: np.dot(x,y)
    svm_model = SVM(lin_kernel, C = i)
    svm_model.fit(X_syn_train, y_syn_train)

    train_pred = svm_model.predict(X_syn_train)
    d1_train_error.append(utils.classification_error(train_pred, y_syn_train))
    test_pred = svm_model.predict(X_syn_test)
    d1_test_error.append(utils.classification_error(test_pred, y_syn_test))

d1_avg_error = []

for i in C_list:
    error_sum = 0

    svm_model1 = SVM(lin_kernel, C = i)
    svm_model2 = SVM(lin_kernel, C = i)
    svm_model3 = SVM(lin_kernel, C = i)
    svm_model4 = SVM(lin_kernel, C = i)
    svm_model5 = SVM(lin_kernel, C = i)

    lo_data_1, tr_data_1 = utils.partition_cross_validation_fold(synthetic_folds, 0)
    lo_data_2, tr_data_2 = utils.partition_cross_validation_fold(synthetic_folds, 1)
    lo_data_3, tr_data_3 = utils.partition_cross_validation_fold(synthetic_folds, 2)
    lo_data_4, tr_data_4 = utils.partition_cross_validation_fold(synthetic_folds, 3)
    lo_data_5, tr_data_5 = utils.partition_cross_validation_fold(synthetic_folds, 4)

    (X_lo1, y_lo1) = lo_data_1
    (X_tr1, y_tr1) = tr_data_1
    svm_model1.fit(X_tr1, y_tr1)
    y_lo_pre1 = svm_model1.predict(X_lo1)
    error1 = utils.classification_error(y_lo_pre1, y_lo1)

    (X_lo2, y_lo2) = lo_data_2
    (X_tr2, y_tr2) = tr_data_2
    svm_model2.fit(X_tr2, y_tr2)
    y_lo_pre2 = svm_model2.predict(X_lo2)
    error2 = utils.classification_error(y_lo_pre2, y_lo2)

    (X_lo3, y_lo3) = lo_data_3
    (X_tr3, y_tr3) = tr_data_3
    svm_model3.fit(X_tr3, y_tr3)
    y_lo_pre3 = svm_model3.predict(X_lo3)
    error3 = utils.classification_error(y_lo_pre3, y_lo3)

    (X_lo4, y_lo4) = lo_data_4
    (X_tr4, y_tr4) = tr_data_4
    svm_model4.fit(X_tr4, y_tr4)
    y_lo_pre4 = svm_model1.predict(X_lo4)
    error4 = utils.classification_error(y_lo_pre4, y_lo4)

    (X_lo5, y_lo5) = lo_data_5
    (X_tr5, y_tr5) = tr_data_5
    svm_model5.fit(X_tr5, y_tr5)
    y_lo_pre5 = svm_model5.predict(X_lo5)
    error5 = utils.classification_error(y_lo_pre5, y_lo5)

    error_sum = error1 + error2 + error3 + error4 + error5

    d1_avg_error.append(error_sum / 5.0)




plt.xscale("log")
plt.plot(C_list, d1_train_error, label = "training_error", color = 'forestgreen')
plt.plot(C_list, d1_test_error, label = "test_error", color = 'mediumvioletred')
plt.plot(C_list, d1_avg_error, label = "avg_cv_error", color = 'dodgerblue')
plt.title("Linear SVM: Errors vs. C")
plt.legend(loc = "upper right")
plt.show()



min_index = d1_avg_error.index(min(d1_avg_error))
print("Chosen C: ")
print(C_list[min_index])
print("Corresponding training error: ")
print(d1_train_error[min_index])
print("Corresponding test error: ")
print(d1_test_error[min_index])

model = SVM(lin_kernel, C = C_list[min_index])
model.fit(X_syn_train, y_syn_train)
fig, ax = plot_contours(model, X_syn_test, y_syn_test, n=100)
fig.show()


# polynomial kernel
q_list = [1,2,3,4,5]


poly_train_error = []
poly_test_error = []
poly_avg_error = []
poly_best_C = []

for q in q_list:
    cv_avg_error = []
    poly_kernel = lambda x,y: (np.dot(np.transpose(x), y)+1) ** q
    for c in C_list:
        model = SVM(poly_kernel, C = c)
        model.fit(X_syn_train, y_syn_train)
        
        avg_error = 0
        for i in range(5):
            lo_data, tr_data = utils.partition_cross_validation_fold(synthetic_folds, i)
            (X_lo, y_lo) = lo_data
            (X_tr, y_tr) = tr_data
            model.fit(X_tr, y_tr)
            pred = model.predict(X_lo)
            avg_error += utils.classification_error(pred, y_lo)
        avg_error /= 5
        cv_avg_error.append(avg_error)
    
    min_cv_error = min(cv_avg_error)
    min_index = cv_avg_error.index(min_cv_error)
    best_C = C_list[min_index]
    poly_best_C.append(best_C)

    model = SVM(poly_kernel, C = best_C)
    model.fit(X_syn_train, y_syn_train)
    train_pred = model.predict(X_syn_train)
    test_pred = model.predict(X_syn_test)
    poly_train_error.append(utils.classification_error(train_pred, y_syn_train))
    poly_test_error.append(utils.classification_error(test_pred, y_syn_test))

    avg_error = 0
    for i in range(5):
        lo_data, tr_data = utils.partition_cross_validation_fold(synthetic_folds, i)
        (X_lo, y_lo) = lo_data
        (X_tr, y_tr) = tr_data
        model.fit(X_tr, y_tr)
        pred = model.predict(X_lo)
        avg_error += utils.classification_error(pred, y_lo)
    avg_error /= 5
    poly_avg_error.append(avg_error)




plt.plot(q_list, poly_train_error, label = "training_error", color = 'forestgreen')
plt.plot(q_list, poly_test_error, label = "test_error", color = 'mediumvioletred')
plt.plot(q_list, poly_avg_error, label = "avg_cv_error", color = 'dodgerblue')
plt.title("Polynomial SVM: Errors vs. q")
plt.legend(loc = "lower left")
plt.show()





min_q_error = min(poly_avg_error)
min_index = poly_avg_error.index(min_q_error)
best_q = q_list[min_index]
print("Chosen q: ")
print(best_q)
print("Corresponding best C: ")
print(poly_best_C[min_index])
print("Corresponding training error: ")
print(poly_train_error[min_index])
print("Corresponding test error: ")
print(poly_test_error[min_index])
model = SVM(lambda x,y: (np.dot(np.transpose(x), y)+1)**best_q,C = poly_best_C[min_index])
model.fit(X_syn_train, y_syn_train)
fig, ax = plot_contours(model, X_syn_test, y_syn_test, n=100)
fig.show()


# compute RBF kernel
def rbf_kernel(x, y, gamma):
    return np.exp(-np.dot(x-y,x-y) * gamma)


C_list = [10**(-3), 10**(-2), 10**(-1), 1, 10, 10**(2)]
gamma_list = [10**(-2), 10**(-1), 1, 10, 10**(2)]

gamma_train_error = []
gamma_test_error = []
gamma_avg_error = []
best_C = []

for g in gamma_list:
    cv_avg_error = []
    for c in C_list:
        model = SVM(lambda x,y: rbf_kernel(x,y,g), C = c)
        model.fit(X_syn_train, y_syn_train)
        
        sum_error = 0
        for i in range(5):
            lo_data, tr_data = utils.partition_cross_validation_fold(synthetic_folds, i)
            (X_lo, y_lo) = lo_data
            (X_tr, y_tr) = tr_data
            model.fit(X_tr, y_tr)
            cv_pred = model.predict(X_lo)
            sum_error += utils.classification_error(cv_pred, y_lo)

        cv_avg_error.append(sum_error / 5)
    
    min_cv_error = min(cv_avg_error)
    min_index = cv_avg_error.index(min_cv_error)
    best_C.append(C_list[min_index])

    model = SVM(lambda x,y: rbf_kernel(x,y,g),C_list[min_index])
    model.fit(X_syn_train, y_syn_train)
    train_pred = model.predict(X_syn_train)
    test_pred = model.predict(X_syn_test)
    gamma_train_error.append(utils.classification_error(train_pred, y_syn_train))
    gamma_test_error.append(utils.classification_error(test_pred, y_syn_test))

    sum_error = 0
    for i in range(5):
        lo_data, tr_data = utils.partition_cross_validation_fold(synthetic_folds, i)
        (X_lo, y_lo) = lo_data
        (X_tr, y_tr) = tr_data
        model.fit(X_tr, y_tr)
        cv_pred = model.predict(X_lo)
        sum_error += utils.classification_error(cv_pred, y_lo)

    gamma_avg_error.append(sum_error / 5)



plt.xscale("log")
plt.plot(gamma_list, gamma_train_error, label = "training_error", color = 'forestgreen')
plt.plot(gamma_list, gamma_test_error, label = "test_error", color = 'mediumvioletred')
plt.plot(gamma_list, gamma_avg_error, label = "avg_cv_error", color = 'dodgerblue')
plt.plot()
plt.title("RBF SVM: Errors vs. Gamma")
plt.legend(loc = "upper left")
plt.show()




min_gamma_error = min(gamma_avg_error)
min_index = gamma_avg_error.index(min_gamma_error)
best_gamma = gamma_list[min_index]
print("gamma chosen: ")
print(best_gamma)
print("Corresponding best C: ")
print(gamma_best_C[min_index])
print("Corresponding training error: ")
print(gamma_train_error[min_index])
print("Corresponding test error: ")
print(gamma_test_error[min_index])
model = SVM(lambda x,y: rbf_kernel(x,y,best_gamma),C = gamma_best_C[min_index])
model.fit(X_syn_train, y_syn_train)
fig, ax = plot_contours(model, X_syn_test, y_syn_test, n=100)
fig.show()


X_train, y_train, X_test, y_test = utils.load_all_train_test_data("/Users/ceciliz/Desktop/ps2_kit/P1/Spam-Dataset")
real_folds = utils.load_all_cross_validation_data("/Users/ceciliz/Desktop/ps2_kit/P1/Spam-Dataset/CrossValidation")

C_list = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 1, 10, 10**(2)]

d2_train_error = []
d2_test_error = []
d2_avg_error = []

lin_kernel = lambda x,y: np.dot(x,y)

for c in C_list:
    model = SVM(lin_kernel, C = c)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    d2_train_error.append(utils.classification_error(train_pred, y_train))
    d2_test_error.append(utils.classification_error(test_pred, y_test))

    sum_error = 0
    for i in range(5):
        lo_data, tr_data = utils.partition_cross_validation_fold(real_folds, i)
        (X_lo, y_lo) = lo_data
        (X_tr, y_tr) = tr_data
        model.fit(X_tr, y_tr)
        pred = model.predict(X_lo)
        sum_error += utils.classification_error(pred, y_lo)
  
    d2_avg_error.append(sum_error / 5)




plt.xscale("log")
plt.plot(C_list, d2_train_error, label = "training_error", color = 'forestgreen')
plt.plot(C_list, d2_test_error, label = "test_error", color = 'mediumvioletred')
plt.plot(C_list, d2_avg_error, label = "avg_cv_error", color = 'dodgerblue')
plt.plot()
plt.title("Linear SVM: Errors vs. C for real data")
plt.legend(loc = "upper right")
plt.show()



min_index = d2_avg_error.index(min(d2_avg_error))
print("C chosen: ")
print(C_list[min_index])
print("Corresponding training error: ")
print(d2_train_error[min_index])
print("Corresponding test error: ")
print(d2_test_error[min_index])


C_list = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 1, 10, 10**(2)]
q_list = [1,2,3,4,5]

poly_train_error = []
poly_test_error = []
poly_avg_error = []
poly_best_C = []

for q in q_list:
    cv_avg_error = []
    poly_kernel = lambda x,y: np.dot(np.transpose(x), y) ** q + 1
    for c in C_list:
        model = SVM(poly_kernel ,C = c)
        model.fit(X_train, y_train)
        
        sum_error = 0
        for i in range(5):
            lo_data, tr_data = utils.partition_cross_validation_fold(real_folds, i)
            (X_lo, y_lo) = lo_data
            (X_tr, y_tr) = tr_data
            model.fit(X_tr, y_tr)
            pred = model.predict(X_lo)
            sum_error += utils.classification_error(pred, y_lo)

        cv_avg_error.append(sum_error / 5)
    
    min_index = cv_avg_error.index(min(cv_avg_error))
    poly_best_C.append(C_list[min_index])

    model = SVM(poly_kernel, C = C_list[min_index])
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    poly_train_error.append(utils.classification_error(train_pred, y_train))
    poly_test_error.append(utils.classification_error(test_pred, y_test))

    sum_error = 0
    for i in range(5):
        lo_data, tr_data = utils.partition_cross_validation_fold(real_folds, i)
        (X_lo, y_lo) = lo_data
        (X_tr, y_tr) = tr_data
        model.fit(X_tr, y_tr)
        pred = model.predict(X_lo)
        sum_error += utils.classification_error(pred, y_lo)
    
    poly_avg_error.append(sum_error / 5)



plt.plot(q_list, poly_train_error, label = "training_error", color = 'forestgreen')
plt.plot(q_list, poly_test_error, label = "test_error", color = 'mediumvioletred')
plt.plot(q_list, poly_avg_error, label = "avg_cv_error", color = 'dodgerblue')
plt.plot()
plt.title("Polynomial SVM: Errors vs. q for real data")
plt.legend(loc = "upper left")
plt.show()




min_index = poly_avg_error.index(min(poly_avg_error))
print("chosen q: ")
print(q_list[min_index])
print("Corresponding best C: ")
print(poly_best_C[min_index])
print("Corresponding training error: ")
print(poly_train_error[min_index])
print("Corresponding test error: ")
print(poly_test_error[min_index])



C_list = [10**(-3), 10**(-2), 10**(-1), 1, 10, 10**(2)]
gamma_list = [10**(-2), 10**(-1), 1, 10, 10**(2)]

gamma_train_error = []
gamma_test_error = []
gamma_avg_error = []
best_C = []

for g in gamma_list:
    cv_avg_error = []
    for c in C_list:
        model = SVM(lambda x,y: compute_RBF(x,y,g), C = c)
        model.fit(X_train, y_train)
        
        sum_error = 0
        for i in range(5):
            lo_data, tr_data = utils.partition_cross_validation_fold(real_folds, i)
            (X_lo, y_lo) = lo_data
            (X_tr, y_tr) = tr_data
            model.fit(X_tr, y_tr)
            pred = model.predict(X_lo)
            sum_error += utils.classification_error(pred, y_lo)

        cv_avg_error.append(sum_error / 5)
    
    min_index = cv_avg_error.index(min(cv_avg_error))
    best_C.append(C_list[min_index])

    model = SVM(lambda x,y: compute_RBF(x,y,g),C = C_list[min_index])
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    gamma_train_error.append(utils.classification_error(train_pred, y_train))
    gamma_test_error.append(utils.classification_error(test_pred, y_test))

    sum_error = 0
    for i in range(5):
        lo_data, tr_data = utils.partition_cross_validation_fold(real_folds, i)
        (X_lo, y_lo) = lo_data
        (X_tr, y_tr) = tr_data
        model.fit(X_tr, y_tr)
        pred = model.predict(X_lo)
        sum_error += utils.classification_error(pred, y_lo)
    
    gamma_avg_error.append(sum_error / 5)



plt.xscale("log")
plt.plot(gamma_list, gamma_train_error, label = "training_error", color = 'forestgreen')
plt.plot(gamma_list, gamma_test_error, label = "test_error", color = 'mediumvioletred')
plt.plot(gamma_list, gamma_avg_error, label = "avg_cv_error", color = 'dodgerblue')
plt.plot()
plt.title("RBF SVM: Errors vs. Gamma for real data")
plt.legend(loc = "upper left")
plt.show()



min_index = gamma_avg_error.index(min(gamma_avg_error))
print("Chosen gamma: ")
print(gamma_list[min_index])
print("Corresponding best C: ")
print(gamma_best_C[min_index])
print("Corresponding training error: ")
print(gamma_train_error[min_index])
print("Corresponding test error: ")
print(gamma_test_error[min_index])

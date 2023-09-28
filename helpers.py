import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve, precision_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, validation_curve, GridSearchCV
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, RandomForestRegressor, \
    GradientBoostingClassifier, GradientBoostingRegressor

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


def check_df(dataframe, head=5):
    """
    To check the data set, it is a function that shows the shape, data types, head, tail, NA values and quantiles.

    Parameters
    ----------
        dataframe: dataframe
                    Data to be checked
        head: int, optional
                    Number of observations to be displayed. Default is 5.

    Returns
    -------
        None

    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def grab_col_names(df, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        df: df
                Değişken isimleri alınmak istenilen df
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and
                   df[col].dtypes != "O"]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and
                   df[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Shows the lower and upper limits of the variable to be examined.

    Parameters
    ----------
        dataframe: dataframe
                    Data to be checked
        col_name: str
                    The name of the variable to be examined for outliers.
        q1: float, optional
                    The value of the first quartile. The default is 0.25.
        q3: float, optional
                    The value of the third quartile. The default is 0.75.

    Returns
    -------
        low_limit: float
                    The lower limit value of the variable.
        up_limit: float
                    The upper limit value of the variable.
        
    Examples
    -------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(outlier_thresholds(df, "sepal_length"))

    Notes   
    -------
        The lower and upper limits of the variable are calculated by the formula below.
        IQR = q3 - q1
        lower_limit = q1 - 1.5 * IQR
        upper_limit = q3 + 1.5 * IQR

    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    """

    Checks whether there is an outlier in the variable to be examined.

    Parameters
    ----------
        dataframe: dataframe
                    Data to be checked
        col_name: str
                    The name of the variable to be examined for outliers.
        q1: float, optional
                    The value of the first quartile. The default is 0.25.
        q3: float, optional
                    The value of the third quartile. The default is 0.75.

    Returns
    -------
        True: bool
                    If there is an outlier, it returns True.
        False: bool
                    If there is no outlier, it returns False.

    Examples
    -------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(check_outlier(df, "sepal_length"))

    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    """

    Replaces outliers with lower and upper limits.

    Parameters
    ----------
        dataframe: dataframe
                    Data to be checked
        variable: str
                    The name of the variable to be examined for outliers.
        q1: float, optional
                    The value of the first quartile. The default is 0.25.
        q3: float, optional
                    The value of the third quartile. The default is 0.75.

    Returns
    -------
        None

    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def missing_values_table(dataframe, na_name=False):
    """

    Shows the number and percentage of missing values in the data set.

    Parameters
    ----------
        dataframe: dataframe
                    Data to be checked
        na_name: bool, optional
                    If True, it returns the names of the variables with missing values. The default is False.

    Returns
    -------
        na_columns: list
                    If na_name=True, it returns the names of the variables with missing values.
        
    Examples
    -------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(missing_values_table(df, na_name=True))

    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    """

    Shows the relationship of the variables with missing values with the target variable.

    Parameters
    ----------
        dataframe: dataframe
                    Data to be checked
        target: str
                    The name of the target variable.
        na_columns: list
                    The names of the variables with missing values.

    Returns
    -------
        None

    Examples
    -------
        import seaborn as sns
        df = sns.load_dataset("iris")
        na_columns = missing_values_table(df, na_name=True)
        missing_vs_target(df, "sepal_length", na_columns)

    """
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    """

    Shows the variables with a correlation value above the threshold.

    Parameters
    ----------
        dataframe: dataframe
                    Data to be checked
        plot: bool, optional
                    If True, it shows the correlation matrix. The default is False.
        corr_th: float, optional
                    Correlation threshold value. The default is 0.90.

    Returns
    -------
        drop_list: list
                    List of variables to be dropped.
        
                    
    Examples
    -------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(high_correlated_cols(df, plot=True))

    """
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu", annot=True)
        plt.show(block=True)
    return drop_list


def label_encoder(dataframe, binary_col):
    """

    Converts binary categorical variables to 0 and 1.

    Parameters
    ----------
        dataframe: dataframe
                    Data to be checked
        binary_col: str, list
                    The name of the binary variables.

    Returns
    -------
        None

    """

    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def rare_analyser(dataframe, target, cat_cols):
    """

    Parameters
    ----------
        dataframe: dataframe
                    Data to be checked
        target: str
                    The name of the target variable.
        cat_cols: list
                    The names of the categorical variables.

    Returns
    -------
        None

    """
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc, cat_cols):
    """

    Parameters
    ----------
        dataframe: dataframe
                    Data to be checked
        rare_perc: float
                    Threshold value for rare variables.
        cat_cols: list
                    The names of the categorical variables.

    Returns
    -------
        dataframe: dataframe

    """
    temp_df = dataframe.copy()
    rare_columns = [col for col in cat_cols if (temp_df[col].value_counts() / len(temp_df) < rare_perc).sum() > 1]

    for col in rare_columns:
        tmp = temp_df[col].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[col] = np.where(temp_df[col].isin(rare_labels), 'Rare', temp_df[col])

    return temp_df


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """

    Encodes categorical variables with one-hot encoding.

    Parameters
    ----------
        dataframe: dataframe
                    Data to be checked
        categorical_cols: list
                    The names of the categorical variables.
        drop_first: bool, optional
                    If True, it creates k-1 dummies. The default is False.

    Returns
    -------
        dataframe: dataframe

    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def plot_importance(model, features, num=5, save=False):
    """

    Parameters
    ----------
        model: object
                    Model object
        features: dataframe
                    Data to be checked
        num: int, optional
                    Number of features to be displayed. The default is 5.
        save: bool, optional
                    If True, it saves the graph. The default is False.

    Returns
    -------
        Figure

    """
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    """

    Parameters
    ----------
        model: object
                    Model object
        X: dataframe
        y: dataframe
        param_name: str
                    The name of the parameter to be examined.
        param_range: list
                    The range of the parameter to be examined.
        scoring: str, optional
                    The name of the metric to be used. The default is "roc_auc".
        cv: int, optional
                    Number of cross validations. The default is 10.

    Returns
    -------


    """
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)


def create_pca_df(X, y):
    """

    Parameters
    ----------
        X: dataframe
        y: dataframe

    Returns
    -------
        final_df: dataframe

    """
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df


def plot_pca(dataframe, target):
    """

    Parameters
    ----------
        dataframe: dataframe
        target: str
                    The name of the target variable.

    Returns
    -------
        Figure

    """
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y", "c", "m", "k", "w"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()


def plot_confussion_matrix(y, y_pred):
    """

    Parameters
    ----------
        y: dataframe
        y_pred: dataframe

    Returns
    -------
        Figure

    """
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()


def cost_function(Y, b, w, X):
    """

    Parameters
    ----------
        Y: dataframe
        b: float
                Bias
        w: float
                Weight
        X: dataframe

    Returns
    -------
        mse: float
                Mean squared error

    """
    m = len(Y)
    sse = 0  #  sum of squared errors

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse


def update_weights(Y, b, w, X, learning_rate):
    """

    Parameters
    ----------
        Y: dataframe
        b: float
                Bias
        w: float
                Weight
        X: dataframe
        learning_rate: float
                Learning rate

    Returns
    -------
        new_b: float
                New bias
        new_w: float
                New weight

    """

    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]

    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)

    return new_b, new_w


def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    """

    Parameters
    ----------
        Y: dataframe
        initial_b: float
                Initial bias
        initial_w: float
                Initial weight
        X: dataframe
        learning_rate: float
                Learning rate
        num_iters: int
                Number of iterations

    Returns
    ------- 
        cost_history: list
                List of cost values
        b: float
                Bias
        w: float
                Weight

    """
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                            cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)

        if i % 100 == 0:  #  print every 100 iterations
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))

    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


# def base_models(X, y, scoring="roc_auc"):
#     print("Base Models....")
#     classifiers = [('LR', LogisticRegression()),
#                    ('KNN', KNeighborsClassifier()),
#                    # ("SVC", SVC()),
#                    ("CART", DecisionTreeClassifier()),
#                    ("RF", RandomForestClassifier()),
#                    ('XGBoost', XGBClassifier(eval_metric='logloss')),
#                    ('LightGBM', LGBMClassifier()),
#                    ('CatBoost', CatBoostClassifier(verbose=False))
#                    ]

#     for name, classifier in classifiers:
#         cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
#         print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


# def hyperparameter_optimization(X, y, cv=10, scoring="roc_auc"):
#     print("Hyperparameter Optimization....")
#     best_models = {}
#     start_time = datetime.now()
#     for name, classifier, params in classifiers:
#         print(f"########## {name} ##########")
#         cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
#         print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

#         gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
#         final_model = classifier.set_params(**gs_best.best_params_)

#         cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
#         print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
#         print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
#         end_time = datetime.now()
#         print(f"Process Time: {end_time - start_time}", end="\n\n")
#         best_models[name] = final_model

#     return best_models


# def voting_classifier(best_models, X, y, cv=3):
#     print("Voting Classifier...")

#     voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
#                                               ('CatBoost', best_models["CatBoost"]),
#                                               ('LightGBM', best_models["LightGBM"]),
#                                               # ('XGBoost', best_models["XGBoost"])
#                                               ],
#                                   voting='soft').fit(X, y)

#     cv_results = cross_validate(voting_clf, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
#     print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
#     print(f"F1Score: {cv_results['test_f1'].mean()}")
#     print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
#     return voting_clf

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# read data
df = pd.read_csv("https://raw.githubusercontent.com/erkansirin78/datasets/master/Churn_Modelling.csv")
df=df.iloc[:,1:]



def dataframe_check(dataframe):
    print("############# Data set Info #################")
    print(dataframe.info())
    print("############# Shape ######################")
    print(dataframe.shape)
    print("############## NA Values ##################")
    print(dataframe.isnull().sum())
    print("############# Data set Head ###############")
    print(dataframe.head())
    print("############# Data set Tail ###############")
    print(dataframe.tail())
    print("############# Data set Describe ############")
    print(dataframe.describe().T)
    print("##########################################")


dataframe_check(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                Class threshold value for numeric but categorical variables
        car_th: int, optinal
                Class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numerical variable list
        cat_but_car: list
                Categorical view cardinal variable list

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = the total number of variables
        num_but_cat is inside cat_cols.
        The sum of 3 lists with return is equal to the total number of variables: cat_cols + num_cols + cat_but_car = number of variables

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car =grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("\n")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("\n")
    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")

for col in num_cols:
    num_summary(df, col)



# def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
#     corr = dataframe.corr()
#     cor_matrix = corr.abs()
#     upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
#     drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
#     if plot:
#         import seaborn as sns
#         import matplotlib.pyplot as plt
#         sns.set(rc={'figure.figsize': (15, 15)})
#         sns.heatmap(corr, cmap="RdBu")
#         plt.show()
#     return drop_list

# high_correlated_cols(df, plot=True)


# def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
#     quartile1 = dataframe[col_name].quantile(q1)
#     quartile3 = dataframe[col_name].quantile(q3)
#     interquantile_range = quartile3 - quartile1
#     up_limit = quartile3 + 1.5 * interquantile_range
#     low_limit = quartile1 - 1.5 * interquantile_range
#     return low_limit, up_limit

# def replace_with_thresholds(dataframe, variable):
#     low_limit, up_limit = outlier_thresholds(dataframe, variable)
#     dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
#     dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
#     dataframe=dataframe[(dataframe[variable]>low_limit)&(dataframe[variable]<up_limit)]

# for i in df.columns:
#     low, up =outlier_thresholds(df, i)
#     if ((df[i] < low) | (df[i] > up)).any():
#         print(f"\nIndices: {df[(df[i] < low) | (df[i] > up)].index}\n")
#         print(df[(df[i] < low) | (df[i] > up)].head())
#         replace_with_thresholds(df,i)


X = df.iloc[:,2:-1].values
print(X.shape)
print(X[:3])

# Output variable
y = df.iloc[:, -1]
print(y.shape)
print(y[:6])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

number_of_trees = 20


os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000/'

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

experiment_name = "Churn_Exp"
mlflow.set_experiment(experiment_name)

registered_model_name = "RFChurnModel"

trf1 = ColumnTransformer([
        ('ohe_gender_geography', OneHotEncoder(sparse=False, handle_unknown='ignore'),[1,2])
    ], remainder='passthrough')

trf2 = RandomForestClassifier()

pipeline = Pipeline([
    ('trf1', trf1),
    ('trf2', trf2)
])


with mlflow.start_run(run_name="with-reg-rf-sklearn") as run:

    pipeline.fit(X_train, y_train)
    print(X_train[:3])
    y_pred = pipeline.predict(X_test)
    print(y_pred[:3])
    (rmse, mae, r2) = eval_metrics(y_test, y_pred)

    print(f"Random Forest model number of trees: {number_of_trees}")
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("n_estimators", number_of_trees)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(pipeline, "model", registered_model_name=registered_model_name)
    else:
        mlflow.sklearn.log_model(pipeline, "model")


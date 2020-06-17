import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

def cat_num_split(df):
    """
    Description:
    ------------
    take a pandas data frame and base "object" data type
    in pandas split categorical and numerical columns.
    
    Parameters
    ----------
    df : pandas data frame
    
    Returns
    -------
    cat_df : contians all categorcal columns
    num_df : contians all numerical columns
    """
    categorical = []
    numerical = []
    for col in df.columns:
        if df[col].dtype == 'O': # "O" stands for Object
            categorical.append(col)
        else:
            numerical.append(col)
    if len(categorical) + len(numerical) != df.shape[1]: # checking the all columns examine
        raise("#cat and #num is not equal #total columns")
    cat_df = df[categorical].copy()
    num_df = df[numerical].copy()    
    return cat_df, num_df


def null_ratio(df,atmost_null=0.6):
    """
    Description:
    ------------
    Take a data frame and filter the some columns that have 
    null percent cell more than specific value value(atmost_null)

    Parameters
    ----------
    df : pandas data frame
    
    atmost_null : the maximun ratio that a column can be contained  null values. 
    if a column has more null then filtered.
    
    Returns
    -------
    not_null_columns (python list): columns which have not any null cells.
    at_most_columns (python list): columns which have null but not more than "atmost_null" criterion.
    -------
    """
    at_most_columns = []
    not_null_columns = []
    temp = df.isna().sum()/len(df) # pandas series that shows null ratio of ecah column
    for index, value in temp.items():
        if 0 < value < atmost_null:
                at_most_columns.append(index)
        elif value == float(0):
            not_null_columns.append(index)    
    return not_null_columns, at_most_columns


def univariant_importance(est,df,target,atmost_null=0.5):
    """
    Description:
    ------------
    Take a scikit learn predictor(estimator) and, bulid simple model
    base on actual value of **single** column and taget to find out 
    the importance of these columns, fianaly we have a report.
    
    Parameters
    ----------
    est: scikit learn predictor. It must choose from a class in scikit learn
    that contian "fit" and "predict" mehtods.
    
    df:pandas data frame. Be curefull you must remove the target column from df.
    
    target: columns of labels.
    
    atmost_null : the maximun ratio that a column can be contained  null values. 
    if a column has more null then filtered.

    Returns
    -------
    out: a pandas data frame that has four columns:
        "col": the name of the column.
        "#": the number of non-null values in which this column has it.
        "accuracy": the accuracy which the predictor find base of "cross_val_score" with 10 folds(cv=10).
        "[#-1 , #+1]": the numbers class member.
        "majority": the ratio of majority class to total non-null values.
    """
    _, cols = null_ratio(df,atmost_null)
    report=[]
    for col in cols:
        mask = df[col].isna().values
        X = df[col][~mask].values
        y = target[~mask].values
        nuiq = np.unique(y,return_counts=True)
        postive_ratio = nuiq[1][1]/(nuiq[1][1]+nuiq[1][0])
        negative_ratio = 1-postive_ratio
        majority = postive_ratio if  postive_ratio > negative_ratio else  negative_ratio
        score = np.mean(cross_val_score(est,X.reshape(-1,1),y,cv=10))
        report.append((col,len(X),score,nuiq[1],majority))
    out = pd.DataFrame(report,columns=["col","#","accuracy", "[#-1 , #+1]","majority"])
    return out


def select_columns(report,thershold=0.05):
    """
    Description:
    ------------
    It takes the report data frame from *univaiant_importance* function and
    selects the important columns from which have null values. I say a feature(column)
    is important if it can predict accuracy more the "threshold" from a random 
    predictor(I find it from "majority" column in report data frame).

    Parameters
    ----------
    report: report data frame from *univaiant_importance* function.
    thershold: the criterion for selecting informative columns
    
    Returns
    -------
    selected_cloumns: columns that select from the columns that 
    at least has one null values and has useful information
    """
    
    selected_cloumns= []
    for _, row in report.iterrows():
        if row[2]-row[4]>thershold:
            selected_cloumns.append(row[0])
    return selected_cloumns

def sample_imputer(df, selected_cloumns):
    """
    Description:
    ------------
    Take sample from non-null value and fill the null cell
    with these sample. (I find out it better and mean and madian)
    
    Parameters
    ----------
    df: pandas data frame
    selected_cloumns: the columns that achive from the *select_columns* function
    
    Returns
    -------
    df : The data frame that null cells of selected columns are filled
    """
    df = df.copy()
    for col in selected_cloumns:
        is_null = df[col].isna()
        null_num = is_null.sum()
        if null_num > 0:
            df.loc[is_null, col] = df[col][~is_null].sample(n=null_num, replace=True).values
    return df

def proba_to_pred(proba, threshold=0.5):
    """
    Description:
    ------------
    Take the probabilty of postive calss and return the hard classifiaction

    Parameters
    ----------
    proba: obabilty of postive calss.
    threshold : make decsion to assign to postive class
    
    Returns
    -------
    y_pred: the list contain +1 and -1
    """
    y_pred =np.zeros_like(proba[:,1])
    for i, prob in enumerate(proba[:,1]):
        if prob > threshold :
            y_pred[i] = +1
        else:
            y_pred[i] = -1
    return y_pred
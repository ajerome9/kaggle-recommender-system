import numpy as np
import csv

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
    
def test_apk():
    print(apk(range(1,6),[6,4,7,1,2], 2), 0.25)
    print(apk(range(1,6),[1,1,1,1,1], 5), 0.2)
    predicted = list(range(1,21))
    predicted.extend(range(200,600))
    print(apk(range(1,100),predicted, 20), 1.0)

def test_mapk():
    print(mapk([range(1,5)],[range(1,5)],3), 1.0)
    print(mapk([[1,3,4],[1,2,4],[1,3]],
        [range(1,6),range(1,6),range(1,6)], 3), 0.685185185185185)
    print(mapk([range(1,6),range(1,6)],
        [[6,4,7,1,2],[1,1,1,1,1]], 5), 0.26)
    print(mapk([[1,3],[1,2,3],[1,2,3]],
        [range(1,6),[1,1,1],[1,2,1]], 3), 11.0/18)

def read_csv_purchases(fn):
    ret = []
    with open(fn, 'r') as f:
        a = csv.reader(f)
        next(a)
        for row in a:
            ret.append(row[1].split(' ') if row[1] else [])
    return ret
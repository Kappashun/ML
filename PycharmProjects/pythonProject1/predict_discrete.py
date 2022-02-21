import pandas as pd
import math
import numpy as np
import time
"""
Module-solution for predicting discrete target of given dataset
using decision tree

Without scikit, only pandas and math

wine_class.ipynb is notebook to demonstrate module work on 
wine dataset from sklearn module
"""


def Entropy(inp):
    '''
    Function calculating entropy of given list of values
    :param inp: list of values
    :return: entropy
    '''
    values = set(inp)
    l = len(inp)
    probs = {i: sum(map(lambda x : x == i, inp))/l for i in values}
    return sum([-probs[i]*math.log(probs[i], 2) for i in values])


def i_g(l, l1, l2):
    '''
    Function calculating information gain of given list of values
    :param l: undivided list
    :param l1: first list from separation
    :param l2: second list from separation
    :return: information gain
    '''
    return Entropy(l) - Entropy(l1) * (len(l1) / len(l)) - Entropy(l2) * (len(l2) / len(l))


def MaxGain(df):
    '''
    Function finding best separation of given dataset (with max information gain)
    :param df: dataset
    :return: tuple of 3 elements: column to apply condition on, dividing value and information gain
    '''
    m_g = -1
    val = -1
    for col in df.columns:
        if col == 'target':
            continue
        l_s = sorted(df[col])
        length = len(df[col])
        for i in l_s:
            cur_g = i_g(df['target'], df[df[col] <= i]['target'], df[df[col] > i]['target'])
            if m_g < cur_g:
                m_g = cur_g
                max_c = col
                val = i

    return (max_c, val, m_g)


class TreeNode():
    '''
    Class for building decision tree
    '''
    def __init__(self, data, left=None, right=None, st=None, res=None):
        '''

        :param data: dataset
        :param left: left node
        :param right: right node
        :param st: return of MaxGain function, best separation for data
        :param res: if no further separation is needed then target, else None
        '''
        self.data = data
        self.left = left
        self.right = right
        self.st = st
        self.res = res

    def copy(self):
        '''
        Copying tree
        :return: tree node clone
        '''
        new_node = TreeNode(data=self.data, st=self.st, res=self.res)
        if self.left:
            new_node.left = self.left.copy()
        if self.right:
            new_node.right = self.right.copy()
        return new_node


def build_tree(df, depth=None):
    '''
    Function for building decision tree, using TreeNode class
    :param df: dataset
    :param depth: max possible depth of tree
    :return: root node of decision tree
    '''
    node = TreeNode(data=df)
    step = MaxGain(df)
    node.st = step
    if step[2] == 0:
        node.res = node.data['target'].iloc[0]
        return node
    if depth != None:
        if depth == 0:
            return node
        node.left = build_tree(df[df[step[0]] <= step[1]], depth-1)
        node.right = build_tree(df[df[step[0]] > step[1]], depth-1)
    else:
        node.left = build_tree(df[df[step[0]] <= step[1]])
        node.right = build_tree(df[df[step[0]] > step[1]])
    return node


def PredictDecisionTree(df):
    '''
    Function to predict target of dataset using decision tree
    :param df: dataset
    :return: prediction column
    '''
    root = build_tree(df, 8)
    def predict_row(row):
        nonlocal root
        rt = root.copy()
        while rt:
            if rt.res != None:
                return rt.res
            if row[rt.st[0]] <= rt.st[1]:
                rt = rt.left
            else:
                rt = rt.right
        return 'Error'
    return df.apply(predict_row, axis=1)


def check_prediction(df, predict_f):
    '''
    Function to measure prediction efficiency
    Showing precision, recall, accuracy and required time of given prediction method
    :param df: dataset
    :param predict_f: prediction function
    :return: None
    '''
    print('Prediction method: ', predict_f.__name__, '\n')
    start = time.time()
    df['predict'] = predict_f(df)
    end = time.time()
    data = [[t,
             df[df['target'] == df['predict']][df['target'] == t].shape[0]/df[df['predict'] == t].shape[0]
             if df[df['predict'] == t].shape[0] != 0 else 0,
             df[df['target'] == df['predict']][df['target'] == t].shape[0]/df[df['target'] == t].shape[0]]
            for t in list(df.target.unique())]
    result = pd.DataFrame(data=data, columns=['Target', 'Precision', 'Recall'])
    accuracy = len(df[df['target'] == df['predict']].index) / len(df.index)
    df.drop(columns='predict', inplace=True)
    print(result)
    print('Accuracy: ', round(accuracy*100, 3), '%')
    print('Time spent: ', round(end-start, 5), ' seconds')
    return

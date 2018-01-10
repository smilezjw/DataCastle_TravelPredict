# coding=utf8

from sklearn import tree
import pandas as pd
from pandas import Series, DataFrame
from datetime import datetime


class DecisionTreeClassifier(object):
    def classify(self, trainDF, testDF):
        clf = tree.DecisionTreeClassifier()
        trainData = trainDF.loc[:, ['actionType', 'orderType_history']]
        trainTarget = trainDF['orderType_future']
        testData = testDF.loc[:, ['actionType', 'orderType']]
        clf.fit(trainData, trainTarget)
        testDF['predict_orderType'] = clf.predict_proba(testData)[:, 1]
        print(testDF)
        print('\r\n')
        return testDF

    def loadCSV(self, filePath, dataFrameName):
        df = pd.read_csv(filePath)
        df.name = dataFrameName
        print('shape: ' + df.name + ' ' + str(df.shape))
        print('user num: ' + str(df['userid'].unique().shape))
        print(df.head(3))
        print('\r\n')
        return df

    # 0 - ordinary order, 1 - quality order
    def loadOrderHistory(self, filePath, dataFrameName=None):
        df = pd.read_csv(filePath)
        userid = df['userid']
        print('OrderHistory shape: ' + str(df.shape))
        print('OrderHistory userid num: ' + str(userid.unique().shape))
        grouped = df.sort_values(['userid', 'orderTime'], ascending=False).groupby('userid').head(1)
        lastOrderType = grouped.loc[:, ['userid', 'orderType']]
        lastOrderType.name = dataFrameName
        print('LastOrderType shape: ' + str(lastOrderType.shape))
        print(lastOrderType[:5])
        print('\r\n')
        return lastOrderType

    def loadOrderFuture(self, filePath, dataFrameName=None):
        df = pd.read_csv(filePath)
        df.name = dataFrameName
        userid = df['userid']
        print(df.head(3))
        print('OrderFuture shape: ' + str(df.shape))
        print('OrderFuture userid num: ' + str(userid.unique().shape))
        print('\r\n')
        return df

    def leftJoinDataFrame(self, leftDF, rightDF, lsuffix=None, rsuffix=None, dataFrameName=None):
        joinedDF = leftDF.merge(rightDF, left_on='userid', right_on='userid', how='left', suffixes=(lsuffix, rsuffix))
        filledDF = joinedDF.fillna(0).astype(int)
        filledDF.name = dataFrameName
        print(leftDF.name + ' & ' + rightDF.name + 'joined DataFrame shape: ' + str(filledDF.shape))
        print(filledDF.head(5))
        print('\r\n')
        return filledDF

    def loadAction(self, filePath, dataFrameName=None):
        df = pd.read_csv(filePath)
        print('Action shape: ' + str(df.shape))
        print('Action userid num: ' + str(df['userid'].unique().shape))
        lastActionDF = df.sort_values(['userid', 'actionTime'], ascending=False).groupby('userid').head(1).loc[:, ['userid', 'actionType']]
        lastActionDF.name = dataFrameName
        print('LastAction shape: ' + str(lastActionDF.shape))
        print('LastAction userid num: ' + str(lastActionDF['userid'].unique().shape))
        print(lastActionDF.head(5))
        print('\r\n')
        return lastActionDF

    def getFilePath(self, relativePath):
        directory = '/Users/jiawenzhang/百度云同步盘/DataCastle/data/'
        return directory + relativePath


if __name__ == '__main__':
    dtc = DecisionTreeClassifier()

    orderHistoryTrainPath = dtc.getFilePath('trainingset/orderHistory_train.csv')
    orderHistoryTestPath = dtc.getFilePath('test/orderHistory_test.csv')
    orderFutureTrainPath = dtc.getFilePath('trainingset/orderFuture_train.csv')
    orderFutureTestPath = dtc.getFilePath('test/orderFuture_test.csv')
    actionTrainPath = dtc.getFilePath('trainingset/action_train.csv')
    actionTestPath = dtc.getFilePath('test/action_test.csv')
    predictPath = dtc.getFilePath('result/' + datetime.now().strftime('%Y%m%d%H%M%S') + '.csv')
    fullResPath = dtc.getFilePath('result/full_' + datetime.now().strftime('%Y%m%d%H%M%S') + '.csv')
    print(fullResPath)

    print("=============== Train Data ===============")
    orderFutureTrainDF = dtc.loadOrderFuture(orderFutureTrainPath, 'OrderFutureTrain')
    orderHistoryTrainDF = dtc.loadOrderHistory(orderHistoryTrainPath, 'OrderHistoryTrain')
    actionTrainDF = dtc.loadAction(actionTrainPath, 'ActionTrain')

    print("=============== Test Data ===============")
    orderFutureTestDF = dtc.loadOrderFuture(orderFutureTestPath, 'OrderFutureTest')
    orderHistoryTestDF = dtc.loadOrderHistory(orderHistoryTestPath, 'OrderHistoryTest')
    actionTestDF = dtc.loadAction(actionTestPath, 'ActionTest')

    print("=============== Joined ===============")
    joinedOrderTrainDF = dtc.leftJoinDataFrame(orderFutureTrainDF, orderHistoryTrainDF, '_future', '_history', 'JoinedOrderTrain')
    joinedOrderTestDF = dtc.leftJoinDataFrame(orderFutureTestDF, orderHistoryTestDF, '_future', '_history', 'JoinedOrderTest')
    joinedActionTrainDF = dtc.leftJoinDataFrame(joinedOrderTrainDF, actionTrainDF, dataFrameName='JoinedActionTrain')
    joinedActionTestDF = dtc.leftJoinDataFrame(joinedOrderTestDF, actionTestDF, dataFrameName='JoinedActionTest')
    # joinedOrderTrainDF.to_csv(joinedTrainPath)

    print("=============== Classify and Predict ===============")
    resultDF = dtc.classify(joinedActionTrainDF, joinedActionTestDF)
    resultDF.to_csv(fullResPath, index=False)
    resultDF[['userid', 'predict_orderType']].to_csv(predictPath, index=False)

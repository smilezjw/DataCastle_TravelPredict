# coding=utf8

from sklearn import tree
import pandas as pd
from pandas import Series, DataFrame
from datetime import datetime
import xgboost as xgb


class DecisionTreeClassifier(object):
    def classify(self, trainDF, testDF):
        trainData = trainDF.loc[:, ['actionType', 'orderType_history']]
        trainTarget = trainDF['orderType_future']
        testData = testDF.loc[:, ['actionType', 'orderType']]
        testData = testData.rename(index=str, columns={'actionType':'actionType', 'orderType': 'orderType_history'})

        dtrain = xgb.DMatrix(trainData, label=trainTarget)
        dtest = xgb.DMatrix(testData)
        params = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
        plst = params.items()
        bst = xgb.train(plst, dtrain, num_boost_round=10)
        pred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
        testDF['predict_orderType'] = pred
        print('>>>>>>>>>>>>>>> ' + str(testDF))
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
        # grouped = df.sort_values(['userid', 'orderTime'], ascending=False).groupby('userid').head(1)
        # lastOrderType = grouped.loc[:, ['userid', 'orderType']]
        # lastOrderType.name = dataFrameName
        # print('LastOrderType shape: ' + str(lastOrderType.shape))
        # print(lastOrderType[:5])
        # print('\r\n')
        # return lastOrderType

        qualitied = df.loc[df['orderType'] == 1].loc[:, ['userid', 'orderType']].drop_duplicates(keep='first')
        qualitied.name = dataFrameName
        print('Qualitied shape: ' + str(qualitied.shape))
        print('Qualitied userid num: ' + str(qualitied['userid'].unique().shape))
        print(qualitied.head(5))
        return qualitied


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
        lastActionDF.name = 'LastActionType'
        print('LastAction shape: ' + str(lastActionDF.shape))
        print('LastAction userid num: ' + str(lastActionDF['userid'].unique().shape))
        print(lastActionDF.head(5))

        timeInterval = df.join(df.loc[:, ['userid', 'actionTime']]
                                 .sort_values(['userid', 'actionTime'], ascending=True)
                                 .groupby('userid').diff(),
                               rsuffix='_interval')
        timeInterval['mean'] = timeInterval['actionTime_interval'].groupby(timeInterval['userid']).transform('mean')
        timeInterval['var'] = timeInterval['actionTime_interval'].groupby(timeInterval['userid']).transform('var')
        timeInterval['min'] = timeInterval['actionTime_interval'].groupby(timeInterval['userid']).transform('min')
        timeInterval = timeInterval.loc[:, ['userid', 'mean', 'var', 'min']].drop_duplicates(keep='first')
        timeInterval.name = 'TimeStatistics'
        print('ActionTime Interval: ' + str(timeInterval.shape))
        print(timeInterval.head(5))
        print('\r\n')
        actionDF = self.leftJoinDataFrame(timeInterval, lastActionDF, dataFrameName=dataFrameName)
        return actionDF

    def loadUserProfile(self, filePath, dataFrameName=None):
        df = pd.read_csv(filePath)
        print('UserProfile shape: ' + str(df.shape))
        print('USerProfile userid num: ' + str(df['userid'].unique().shape))
        userProfileDF = df.loc[:, ['userid', 'age']]
        userProfileDF['age'] = userProfileDF['age'].replace('后', '', regex=True).fillna(0).astype(int)
        userProfileDF.name = dataFrameName
        print(userProfileDF.head(5))
        print('\r\n')
        return userProfileDF

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
    userProfileTrainPath = dtc.getFilePath('trainingset/userProfile_train.csv')
    userProfileTestPath = dtc.getFilePath('test/userProfile_test.csv')
    predictPath = dtc.getFilePath('result/' + datetime.now().strftime('%Y%m%d%H%M%S') + '.csv')
    fullResPath = dtc.getFilePath('result/full_' + datetime.now().strftime('%Y%m%d%H%M%S') + '.csv')
    print(fullResPath)

    print("=============== Train Data ===============")
    orderFutureTrainDF = dtc.loadOrderFuture(orderFutureTrainPath, 'OrderFutureTrain')
    orderHistoryTrainDF = dtc.loadOrderHistory(orderHistoryTrainPath, 'OrderHistoryTrain')
    actionTrainDF = dtc.loadAction(actionTrainPath, 'ActionTrain')
    userProfileTrainDF = dtc.loadUserProfile(userProfileTrainPath, 'UserProfileTrain')

    print("=============== Test Data ===============")
    orderFutureTestDF = dtc.loadOrderFuture(orderFutureTestPath, 'OrderFutureTest')
    orderHistoryTestDF = dtc.loadOrderHistory(orderHistoryTestPath, 'OrderHistoryTest')
    actionTestDF = dtc.loadAction(actionTestPath, 'ActionTest')
    userProfileTestDF = dtc.loadUserProfile(userProfileTestPath, 'UserProfileTest')


    print("=============== Joined ===============")
    joinedOrderTrainDF = dtc.leftJoinDataFrame(orderFutureTrainDF, orderHistoryTrainDF, '_future', '_history', 'JoinedOrderTrain')
    joinedOrderTestDF = dtc.leftJoinDataFrame(orderFutureTestDF, orderHistoryTestDF, '_future', '_history', 'JoinedOrderTest')
    joinedActionTrainDF = dtc.leftJoinDataFrame(joinedOrderTrainDF, actionTrainDF, dataFrameName='JoinedActionTrain')
    joinedActionTestDF = dtc.leftJoinDataFrame(joinedOrderTestDF, actionTestDF, dataFrameName='JoinedActionTest')
    joinedUserProfileTrainDF = dtc.leftJoinDataFrame(joinedActionTrainDF, userProfileTrainDF, dataFrameName='JoinedUserProfileTrain')
    joinedUserProfileTestDF = dtc.leftJoinDataFrame(joinedActionTestDF, userProfileTestDF, dataFrameName='JoinedUserProfileTest')

    print("=============== Classify and Predict ===============")
    resultDF = dtc.classify(joinedUserProfileTrainDF, joinedUserProfileTestDF)
    resultDF.to_csv(fullResPath, index=False)
    resultDF[['userid', 'predict_orderType']].to_csv(predictPath, index=False)

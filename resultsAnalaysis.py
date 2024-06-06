import json
import numpy as np
from scipy import stats

############################
# Calculate the mean of 100 sets of random seed data.
############################


# Read macro avg data from a json file into the corresponding array.
def getFlData(basePath, seed, fileName):

    with open(f"{basePath}/seed={seed}/{fileName}", "r") as file:
        dataList = json.load(file)
        file.close()

    for data in dataList:
        if data['num_of_comm'] == 350:  # Only the data when the newsletter round is 350 is needed. Commenting out is averaging out all rounds
            FL_AccuracyList.append(data['accuracy'])
            FL_PrecisionList.append(data['macro avg']['precision'])
            FL_RecallList.append(data['macro avg']['recall'])
            FL_F1ScoreList.append(data['macro avg']['f1-score'])
    # print(len(FL_AccuracyList))

basePath = './split-0.3-seedRange(1,100)'


# Array where the experimental data is stored in the json file --AvgFL
FL_AccuracyList = []  # Location where the three client accuracies are stored in FL
FL_PrecisionList = []
FL_RecallList = []
FL_F1ScoreList = []

for i in range(100):
    getFlData(basePath, seed=i+1, fileName="client1.json")
    getFlData(basePath, seed=i+1, fileName="client2.json")
    getFlData(basePath, seed=i+1, fileName="client3.json")

print("AVG_FL_Accuracy:\t", sum(FL_AccuracyList)/len(FL_AccuracyList))
# print("len(FL_AccuracyList) = ", len(FL_AccuracyList))
print("AVG_FL_Precision:\t", sum(FL_PrecisionList)/len(FL_PrecisionList))
print("AVG_FL_Recall:\t", sum(FL_RecallList)/len(FL_RecallList))
print("AVG_FL_F1Score:\t", sum(FL_F1ScoreList)/len(FL_F1ScoreList))

print("AVG_FL_Accuracy_95%:\t", stats.norm.interval(0.95, loc=np.mean(FL_AccuracyList), scale=stats.sem(FL_AccuracyList)))
print("AVG_FL_Precision_95%:\t", stats.norm.interval(0.95, loc=np.mean(FL_PrecisionList), scale=stats.sem(FL_PrecisionList)))
print("AVG_FL_Recall_95%:\t", stats.norm.interval(0.95, loc=np.mean(FL_RecallList), scale=stats.sem(FL_RecallList)))
print("AVG_FL_F1Score_95%:\t", stats.norm.interval(0.95, loc=np.mean(FL_F1ScoreList), scale=stats.sem(FL_F1ScoreList)))

##################################
# DITM平均
##################################

# Array where the experimental data is stored in the json file  --DITM
DITM_AccuracyList = []  # Location where the three client accuracy values are stored in DITM
DITM_PrecisionList = []
DITM_RecallList = []
DITM_F1ScoreList = []


# Read macro avg data from json file into the corresponding array
def getDitmData(basePath, seed):

    with open(f"{basePath}/seed={seed}/DITM.json", "r") as file:
        data = json.load(file)
        for client in data:
            value = list(client.values())[0]  # get value
            # print(value)
            DITM_AccuracyList.append(value['accuracy'])
            DITM_PrecisionList.append(value['macro avg']['precision'])
            DITM_RecallList.append(value['macro avg']['recall'])
            DITM_F1ScoreList.append(value['macro avg']['f1-score'])
        file.close()

for i in range(100):
    getDitmData(basePath, i + 1)

print("AVG_DITM_Accuracy:\t", sum(DITM_AccuracyList)/len(DITM_AccuracyList))
# print("len(DITM_AccuracyList) = ", len(DITM_AccuracyList))
print("AVG_DITM_Precision:\t", sum(DITM_PrecisionList)/len(DITM_PrecisionList))
print("AVG_DITM_Recall:\t", sum(DITM_RecallList)/len(DITM_RecallList))
print("AVG_DITM_F1Score:\t", sum(DITM_F1ScoreList)/len(DITM_F1ScoreList))

print("AVG_DITM_Accuracy_95%:\t", stats.norm.interval(0.95, loc=np.mean(DITM_AccuracyList), scale=stats.sem(DITM_AccuracyList)))
print("AVG_DITM_Precision_95%:\t", stats.norm.interval(0.95, loc=np.mean(DITM_PrecisionList), scale=stats.sem(DITM_PrecisionList)))
print("AVG_DITM_Recall_95%:\t", stats.norm.interval(0.95, loc=np.mean(DITM_RecallList), scale=stats.sem(DITM_RecallList)))
print("AVG_DITM_F1Score_95%:\t", stats.norm.interval(0.95, loc=np.mean(DITM_F1ScoreList), scale=stats.sem(DITM_F1ScoreList)))
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report
import numpy as np
import random
import os
from Clients_Dy import client1,client2,client3
from GetData_Dy import SpiltData_ex1,SpiltData_ex2,GetDataSet,GetDataSet2,GetDataSet3
from tqdm import tqdm
import warnings
import json


dir = "split-0.3-seedRange(1,100)"
class Neuro_net(torch.nn.Module):
    def __init__(self):
        super(Neuro_net, self).__init__()
        self.layer1 = torch.nn.Linear(40, 20)
        self.layer2 = torch.nn.Linear(20, 10)
        self.layer3 = torch.nn.Linear(10, 5)
        self.layer4 = torch.nn.Linear(5, 2)
        self.layer5 = torch.nn.Softmax(dim=0)

    def forward(self, input):
        tensor = torch.relu(self.layer1(input))
        tensor = torch.relu(self.layer2(tensor))
        tensor = torch.relu(self.layer3(tensor))
        tensor = self.layer4(tensor)
        tensor = self.layer5(tensor)
        return tensor

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


# saving as json file
def saveData(fileName, dataList):
    if not os.path.exists(f"./{dir}/{fileName.split('/')[-2]}"):
        os.mkdir(f"./{dir}/{fileName.split('/')[-2]}")

    with open(fileName, "w") as file:
        json.dump(dataList, file, indent=4)
        file.close()


def main(splitSeed):
    # --------------------------------------------------------------------------#
    num_of_communication = 351  # communication rounds = num_of_communication - 1
    learning_rate = 0.001
    momt = 0.9
    localEpoch = 5
    localBatchSize = 10
    num_of_client = 3
    # --------------------------------------------------------------------------#


    spilt_num = 50
    os.environ['CUDA_VISIBLE_DEVICES'] = 'gpu0'
    mode = 0            # Experimental mode


    # --------------------------------------------------------------------------#

    if mode == 0:
        SD = SpiltData_ex1()
        SD.spilt(splitSeed)

    if mode == 1:
        SD = SpiltData_ex2()
        SD.spilt(spilt_num)

    # --------------------------------------------------------------------------#
    #   Set random seed:
    # --------------------------------------------------------------------------#
    setup_seed(6)  # 1
    # --------------------------------------------------------------------------#

    data_test_h1 = GetDataSet()

    test_h1_x = torch.from_numpy(data_test_h1.test_data).float()
    test_h1_y = torch.from_numpy(data_test_h1.test_label)

    data_test_h2 = GetDataSet2()

    test_h2_x = torch.from_numpy(data_test_h2.test_data).float()
    test_h2_y = torch.from_numpy(data_test_h2.test_label)

    data_test_h3 = GetDataSet3()

    test_h3_x = torch.from_numpy(data_test_h3.test_data).float()
    test_h3_y = torch.from_numpy(data_test_h3.test_label)

    # Start training

    net_h1 = Neuro_net()
    net_h2 = Neuro_net()
    net_h3 = Neuro_net()

    optimizer_h1 = torch.optim.SGD(net_h1.parameters(), lr=learning_rate, momentum=momt)
    optimizer_h2 = torch.optim.SGD(net_h2.parameters(), lr=learning_rate, momentum=momt)
    optimizer_h3 = torch.optim.SGD(net_h3.parameters(), lr=learning_rate, momentum=momt)

    loss_function_h1 = torch.nn.CrossEntropyLoss()
    loss_function_h2 = torch.nn.CrossEntropyLoss()
    loss_function_h3 = torch.nn.CrossEntropyLoss()

    # dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    myClients_h1 = client1(dev)
    myClients_h2 = client2(dev)
    myClients_h3 = client3(dev)

    local_parme_iso1 = {}
    local_parme_iso2 = {}
    local_parme_iso3 = {}

    for key, var in net_h1.state_dict().items():
        local_parme_iso1[key] = var.clone()
        local_parme_iso2[key] = var.clone()
        local_parme_iso3[key] = var.clone()

    # --------------------------------------------------------------------------#
    # Baseline: DITM (tips: DITM is equal to CML)
    #--------------------------------------------------------------------------#
    # ---------------------------------------------------------------------------------------------------------------
    iso_h1_test = {}
    iso_h2_test = {}
    iso_h3_test = {}
    # ---------------------------------------------------------------------------------------------------------------
    order = np.random.permutation(1)
    clients_in_comm = ['client{}'.format(i) for i in order[0:1]]

    for clts in tqdm(clients_in_comm):
        local_parameters_iso1 = myClients_h1.localUpdate1(localEpoch, localBatchSize, net_h1 , loss_function_h1, optimizer_h1, local_parme_iso1)

    for clts in tqdm(clients_in_comm):
        local_parameters_iso2 = myClients_h2.localUpdate2(localEpoch, localBatchSize, net_h2, loss_function_h2, optimizer_h2, local_parme_iso2)

    for clts in tqdm(clients_in_comm):
        local_parameters_iso3 = myClients_h3.localUpdate3(localEpoch, localBatchSize, net_h3, loss_function_h3, optimizer_h3, local_parme_iso3)


    net_h1.load_state_dict(local_parameters_iso1, strict=True)
    net_h2.load_state_dict(local_parameters_iso2, strict=True)
    net_h3.load_state_dict(local_parameters_iso3, strict=True)

    out_iso1 = net_h1(test_h1_x)
    out_iso2 = net_h2(test_h2_x)
    out_iso3 = net_h3(test_h3_x)

    # Stores the output data temporarily so that it can be written to a json file at the end.
    jsonBuffer = []

    print('DITM:')
    pridect_iso1_y = torch.max(out_iso1, dim=1)[1]
    pridect_iso1_label = pridect_iso1_y.data.numpy()
    true_iso1_label = test_h1_y.data.numpy()

    accuracy_iso1 = float((pridect_iso1_label == true_iso1_label).astype(int).sum()) / float(true_iso1_label.size)
    warnings.filterwarnings("ignore")
    print(f'acc_ditm1:{accuracy_iso1}')
    report_iso1 = classification_report(true_iso1_label, pridect_iso1_label, labels=[0, 1],target_names=['Mild', 'Severe'], output_dict=True)
    print(report_iso1)

    # Stored in jsonBuffer waiting for final write to json file, DITM no communication rounds
    jsonBuffer.append({"ditm1": report_iso1})

    # ---------------------------------------------------------------------------------------------------------------
    iso_h1_test['test_h1_y'] = test_h1_y
    iso_h1_test['pridect_h1_label'] = pridect_iso1_label
    # ---------------------------------------------------------------------------------------------------------------

    pridect_iso2_y = torch.max(out_iso2, dim=1)[1]
    pridect_iso2_label = pridect_iso2_y.data.numpy()
    true_iso2_label = test_h2_y.data.numpy()

    accuracy_iso2 = float((pridect_iso2_label == true_iso2_label).astype(int).sum()) / float(true_iso2_label.size)
    warnings.filterwarnings("ignore")
    print(f'acc_ditm2:{accuracy_iso2}')
    report_iso2 = classification_report(true_iso2_label, pridect_iso2_label, labels=[0, 1],target_names=['Mild', 'Severe'], output_dict=True)
    print(report_iso2)

    # Stored in jsonBuffer waiting for final write to json file, DITM no communication rounds
    jsonBuffer.append({"ditm2": report_iso2})

    # ---------------------------------------------------------------------------------------------------------------
    iso_h2_test['test_h2_y'] = test_h2_y
    iso_h2_test['pridect_h2_label'] = pridect_iso2_label
    # ---------------------------------------------------------------------------------------------------------------
    pridect_iso3_y = torch.max(out_iso3, dim=1)[1]
    pridect_iso3_label = pridect_iso3_y.data.numpy()
    true_iso3_label = test_h3_y.numpy()

    accuracy_iso3 = float((pridect_iso3_label == true_iso3_label).astype(int).sum()) / float(true_iso3_label.size)
    warnings.filterwarnings("ignore")
    print(f'acc_ditm3:{accuracy_iso3}')
    report_iso3 = classification_report(true_iso3_label, pridect_iso3_label, labels=[0, 1],target_names=['Mild', 'Severe'], output_dict=True)
    print(report_iso3)

    # Stored in jsonBuffer waiting for final write to json file, DITM no communication rounds
    jsonBuffer.append({"ditm3": report_iso3})

    # ---------------------------------------------------------------------------------------------------------------
    iso_h3_test['test_h3_y'] = test_h3_y
    iso_h3_test['pridect_h3_label'] = pridect_iso3_label
    # ---------------------------------------------------------------------------------------------------------------

    # Save
    saveData(f"./{dir}/seed={splitSeed}/DITM.json", jsonBuffer)
    del jsonBuffer  # Free up storage space

    # --------------------------------------------------------------------------#
    # AVG_FL
    # --------------------------------------------------------------------------#
    print("*" * 30)
    print('AVG_FL:')

    global_parameters = {}

    for key, var in net_h1.state_dict().items():
        global_parameters[key] = var.clone()

    print("*" * 30)

    # Stores the output data temporarily so that it can be written to a json file at the end.
    jsonBuffer1 = []
    jsonBuffer2 = []
    jsonBuffer3 = []

    for num_of_comm in range(num_of_communication):
        order = np.random.permutation(1)
        clients_in_comm = ['client{}'.format(i) for i in order[0:1]]

        for clts in tqdm(clients_in_comm):
            local_parameters_h1 = myClients_h1.localUpdate1(localEpoch, localBatchSize, net_h1, loss_function_h1, optimizer_h1, global_parameters)

        for clts in tqdm(clients_in_comm):
            local_parameters_h2 = myClients_h2.localUpdate2(localEpoch, localBatchSize, net_h2, loss_function_h2, optimizer_h2, global_parameters)

        for clts in tqdm(clients_in_comm):
            local_parameters_h3 = myClients_h3.localUpdate3(localEpoch, localBatchSize, net_h3, loss_function_h3, optimizer_h3, global_parameters)

        for key in global_parameters:
            global_parameters[key] = ((local_parameters_h1[key] + local_parameters_h2[key] + local_parameters_h3[key]) / num_of_client)

        net_h1.load_state_dict(global_parameters, strict=True)
        net_h2.load_state_dict(global_parameters, strict=True)
        net_h3.load_state_dict(global_parameters, strict=True)

        out1 = net_h1(test_h1_x)
        out2 = net_h2(test_h2_x)
        out3 = net_h3(test_h3_x)

        with torch.no_grad():
             if num_of_comm != 0 and num_of_comm % 1 == 0:
                print("num_of_comm: {} ".format(num_of_comm))
                # ---------------------------------------------------------------------------------------------------------------
                h1_test = {}
                h2_test = {}
                h3_test = {}
                # ---------------------------------------------------------------------------------------------------------------
                plt.cla()
                pridect_h1_y = torch.max(out1, dim=1)[1]
                pridect_h1_label = pridect_h1_y.data.numpy()
                true_h1_label = test_h1_y.data.numpy()

                accuracy_h1 = float((pridect_h1_label == true_h1_label).astype(int).sum()) / float(true_h1_label.size)
                warnings.filterwarnings("ignore")
                print(f'acc_h1:{accuracy_h1}')
                report_h1 = classification_report(true_h1_label, pridect_h1_label, labels=[0, 1], target_names=['Mild', 'Severe'], output_dict=True)
                print(report_h1)

                record = {'num_of_comm': num_of_comm}  # Record the rounds
                # one record of json file
                jsonBuffer1.append(dict(**record, **report_h1))  # 1. splicing on the experimental data  2. stored in the jsonBuffer waiting for the final write json file

                # ---------------------------------------------------------------------------------------------------------------
                h1_test['test_h1_y'] = test_h1_y
                h1_test['pridect_h1_label'] = pridect_h1_label
                # ---------------------------------------------------------------------------------------------------------------

                pridect_h2_y = torch.max(out2, dim=1)[1]
                pridect_h2_label = pridect_h2_y.data.numpy()
                true_h2_label = test_h2_y.data.numpy()

                accuracy_h2 = float((pridect_h2_label == true_h2_label).astype(int).sum()) / float(true_h2_label.size)
                warnings.filterwarnings("ignore")
                print(f'acc_h2:{accuracy_h2}')
                report_h2 = classification_report(true_h2_label, pridect_h2_label, labels=[0, 1],target_names=['Mild', 'Severe'], output_dict=True)
                print(report_h2)

                # one record of json file
                jsonBuffer2.append(dict(**record, **report_h2))  # 1. splicing on the experimental data  2. stored in the jsonBuffer waiting for the final write json file

                # ---------------------------------------------------------------------------------------------------------------
                h2_test['test_h2_y'] = test_h2_y
                h2_test['pridect_h2_label'] = pridect_h2_label
                # ---------------------------------------------------------------------------------------------------------------

                pridect_h3_y = torch.max(out3, dim=1)[1]
                pridect_h3_label = pridect_h3_y.data.numpy()
                true_h3_label = test_h3_y.data.numpy()

                accuracy_h3 = float((pridect_h3_label == true_h3_label).astype(int).sum()) / float(true_h3_label.size)
                warnings.filterwarnings("ignore")
                print(f'acc_h3:{accuracy_h3}')
                report_h3 = classification_report(true_h3_label, pridect_h3_label, labels=[0, 1],target_names=['Mild', 'Severe'], output_dict=True)
                print(report_h3)

                # one record of json file
                jsonBuffer3.append(dict(**record, **report_h3))  # 1. splicing on the experimental data  2. stored in the jsonBuffer waiting for the final write json file

                # ---------------------------------------------------------------------------------------------------------------
                h3_test['test_h3_y'] = test_h3_y
                h3_test['pridect_h3_label'] = pridect_h3_label
                # ---------------------------------------------------------------------------------------------------------------


    # data saving
    saveData(f"./{dir}/seed={splitSeed}/client1.json", jsonBuffer1)
    saveData(f"./{dir}/seed={splitSeed}/client2.json", jsonBuffer2)
    saveData(f"./{dir}/seed={splitSeed}/client3.json", jsonBuffer3)


if __name__ == '__main__':
    seeds = 100  # Random seed range to be tested

    if not os.path.exists(f"./{dir}"):
        os.mkdir(f'./{dir}')

    for seed in range(seeds):
        main(seed + 1)  # Tested under 100 different random seeds





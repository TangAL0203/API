#-*-coding:utf-8-*-
import os
import os.path as osp
import torch
from torch.autograd import Variable
from utils.MyBCEWithLogitsLoss import BCEWithLogitsLoss as MyBCEWithLogitsLoss
from tqdm import tqdm

def GetTrueLabel(label):
    index = []
    for s_label in label:
        temp = []
        temp = [m for m,j in enumerate(s_label) if j==1]
        index.append(temp)
    true_label = []
    for i in index:
        temp = []
        for j in i:
            if j in range(0,14):
                temp.append([0, j])
            elif j in range(14,23,1):
                temp.append([1, j-14])
            elif j in range(23,100,1):
                temp.append([2, j-23])
            elif j in range(100,107,1):
                temp.append([3, j-100])
            elif j in range(107,115,1):
                temp.append([4, j-107])
            elif j in range(115,141,1):
                temp.append([5, j-115])
            elif j in range(141,162,1):
                temp.append([6, j-141])
            elif j in range(162,169,1):
                temp.append([7, j-162])
            elif j in range(169,184,1):
                temp.append([8, j-169])
            elif j in range(184,223,1):
                temp.append([9, j-184])
            elif j in range(223,235,1):
                temp.append([10, j-223])
            elif j in range(235,245,1):
                temp.append([11, j-235])
            elif j in range(245,254,1):
                temp.append([12, j-245])
            elif j in range(254,262,1):
                temp.append([13, j-254])
            elif j in range(262,273,1):
                temp.append([14, j-262])
            elif j in range(273,285,1):
                temp.append([15, j-273])
            elif j in range(285,292,1):
                temp.append([16, j-285])
            elif j in range(292,303,1):
                temp.append([17, j-292])
        true_label.append(temp)
    return true_label

########### for sigmoid_cross_entropy_with_logits ##########

# change +1 to torch.FloatTensor(1) and -1 into torch.FloatTensor()
def change_label(label):
    true_label = []
    for s_label in label:
        temp = [1 if i==1 else 0 for i in s_label]
        true_label.append(temp)
        
    return Variable(torch.FloatTensor(true_label).cuda())  #  N X C

#### single branch
def sigmoid_train_batch(model, criterion, optimizer, batch_label):
    optimizer.zero_grad() #
    C = batch_label[0][0].shape[0]
    H = batch_label[0][0].shape[1]
    W = batch_label[0][0].shape[2]
    batch = torch.cat((batch_label[i][0] for i in range(len(batch_label))),0).view(-1,C,H,W)  #  N C H W
    input = Variable(batch.cuda())
    label = [batch_label[i][1] for i in range(len(batch_label))]
    target = change_label(label)
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    return loss.data

def sigmoid_train_epoch(model, num_batches, train_loader, print_freq, optimizer=None):
    criterion = torch.nn.BCEWithLogitsLoss()
    for batch_label in train_loader:
        loss = sigmoid_train_batch(model, criterion, optimizer, batch_label)
        if num_batches%print_freq == 0:
            print('%23s%-9s%-13s'%('the '+str(num_batches)+'th batch, ','loss is: ',str(round(loss[0],8))))
        num_batches +=1
    return num_batches

#### multi branch
# no category branch
Attr_Num_dict = {'category':23, 'length_of_upper_body_clothes':5, 'length_of_trousers':5, 'length_of_dresses':5, 'length_of_sleeves':8,\
'fitness_of_clothes':5, 'design_of_dresses':10, 'type_of_sleeves':10, 'type_of_trousers':7, 'type_of_dresses':12,\
'type_of_collars':10, 'type_of_waistlines':7, 'type_of_clothes_buttons':7, 'thickness_of_clothes':4, 'fabric_of_clothes':20,\
'style_of_clothes':23, 'part_details_of_clothes':72, 'graphic_elements_texture':47}

Attr = ['length_of_upper_body_clothes', 'length_of_trousers', 'length_of_dresses', 'length_of_sleeves', 'fitness_of_clothes',\
'design_of_dresses', 'type_of_sleeves', 'type_of_trousers', 'type_of_dresses', 'type_of_collars', 'type_of_waistlines', \
'type_of_clothes_buttons', 'thickness_of_clothes', 'fabric_of_clothes', 'style_of_clothes', \
'part_details_of_clothes', 'graphic_elements_texture']

# input: labels, Each_Attr_id
# labels = [1,+1,-1,-1,+1,+1,...]*batch_size
# out  branch_ids: [[0,1,2],[1,3],...]  branch_label: [[1,0,1,0,0,0,1..],[1,0,0,1,1,0,...]...]
def get_branch(Each_Attr_id, labels):
    branch_label = []
    branch_ids = []
    index1_index0 = {}
    Attr_ids = []
    for key in Attr:
        Attr_ids.append(Each_Attr_id[key])
    index1 = 0
    for index0, label in enumerate(labels):
        temp = []
        temp1 = []
        cur_ids = []
        for i,j in enumerate(label):
            if j==1:
                for m, ids in enumerate(Attr_ids):
                    if i in ids and m not in temp:
                        temp.append(m)
                        cur_ids += ids   
        if len(cur_ids)!=0:
            index1_index0[index1] = index0 # index0 is smaple id in batch, index1 is the out sample id(has at least one attr)
            branch_ids.append(temp)
            branch_label.append(Variable(torch.unsqueeze(torch.FloatTensor([label[k] for k in cur_ids]), dim=0).cuda()))
            index1+=1

    return index1_index0, branch_ids, branch_label

## only multi label
def multi_task_train_batch(model, Each_Attr_id, criterion, optimizer, batch_label):
    optimizer.zero_grad() #
    C = batch_label[0][0].shape[0]
    H = batch_label[0][0].shape[1]
    W = batch_label[0][0].shape[2]
    batch = torch.cat((batch_label[i][0] for i in range(len(batch_label))),0).view(-1,C,H,W)  #  N C H W
    input = Variable(batch.cuda())

    labels = [batch_label[i][1] for i in range(len(batch_label))]  # [[1,-1..],[1,1,-1..]]
    for i, label in enumerate(labels):
        labels[i] = [1 if j==1 else 0 for j in label]
    index1_index0, branch_ids, branch_label = get_branch(Each_Attr_id, labels)

    output = model(input)  #  [NxC1, NxC2, NxC3, NxC4, NxC5, ..., NxC17]

    loss = []
    for i,(branch_id,label) in enumerate(zip(branch_ids, branch_label)):
        index0 = index1_index0[i]
        if len(branch_id)>=1:
            cur_output = torch.unsqueeze(torch.cat(tuple((output[j][index0] for j in branch_id)),0), dim=0)
        else:
            cur_output = output[branch_id[0]][index0]
        loss.append(criterion(cur_output, label))

    total_loss = sum(loss)/len(loss)
    total_loss.backward() 
    optimizer.step()

    return total_loss.data

def multi_task_train_epoch(model, Each_Attr_id, num_batches, train_loader, print_freq, optimizer=None):
    criterion = torch.nn.BCEWithLogitsLoss()
    for batch_label in train_loader:
        loss = multi_task_train_batch(model, Each_Attr_id, criterion, optimizer, batch_label)
        if num_batches%print_freq == 0:
            print('%23s%-9s%-13s'%('the '+str(num_batches)+'th batch, ','loss is: ',str(round(loss[0],8))))
        num_batches +=1
    return num_batches


## multi_class + multi_label training
# ignore category
# 'length_of_upper_body_clothes':0,
# 'length_of_trousers':0,
# 'length_of_dresses':0,
# 'length_of_sleeves':1,
# 'fitness_of_clothes':1,
# 'design_of_dresses':1,
# 'type_of_sleeves':0,
# 'type_of_trousers':1,
# 'type_of_dresses':0,
# 'type_of_collars':0,
# 'type_of_waistlines':1,
# 'type_of_clothes_buttons':1,
# 'thickness_of_clothes':0,
# 'fabric_of_clothes':1,
# 'style_of_clothes':1,
# 'part_details_of_clothes':1,
# 'graphic_elements_texture':1
def mix_multi_task_train_batch(model, Each_Attr_id, multi_label, multi_class, optimizer, batch_label):
    multi_class_indexs = [0,1,2,6,8,9,12]
    multi_label_indexs = [3,4,5,7,10,11,13,14,15,16]
    num_each_attr = [5, 5, 5, 8, 5, 10, 10, 7, 12, 10, 7, 7, 4, 20, 23, 72, 47]
    if optimizer is None:
        pass
    else:
        optimizer.zero_grad() #
    C = batch_label[0][0].shape[0]
    H = batch_label[0][0].shape[1]
    W = batch_label[0][0].shape[2]
    batch = torch.cat((batch_label[i][0] for i in range(len(batch_label))),0).view(-1,C,H,W)  #  N C H W
    input = Variable(batch.cuda())

    labels = [batch_label[i][1] for i in range(len(batch_label))]  # [[1,-1..],[1,1,-1..]]
    for i, label in enumerate(labels):
        labels[i] = [1 if j==1 else 0 for j in label]
    index1_index0, branch_ids, _ = get_branch(Each_Attr_id, labels)

    output = model(input)  #  [NxC1, NxC2, NxC3, NxC4, NxC5, ..., NxC17]

    loss = []
    for i,(branch_id,label) in enumerate(zip(branch_ids, labels)):
        index0 = index1_index0[i]
        for index in branch_id:
            if index in multi_class_indexs:
                cur_output = torch.unsqueeze(output[index][index0], dim=0)
                if index==0:
                    start = 23
                else:
                    start = sum(num_each_attr[0:index])+23
                end = num_each_attr[index]+start
                for tt,qq in enumerate(label[start:end]):
                    if qq==1:
                        cur_label = Variable(torch.LongTensor([tt])).cuda()
                        loss.append(multi_class(cur_output, cur_label))
                        break
            elif index in multi_label_indexs:
                if index==0:
                    start = 23
                else:
                    start = sum(num_each_attr[0:index])+23
                end = num_each_attr[index]+start
                cur_output = torch.unsqueeze(output[index][index0], dim=0)
                cur_label  = Variable(torch.unsqueeze(torch.FloatTensor(label[start:end]), dim=0).cuda())
                loss.append(multi_label(cur_output, cur_label))
    

    total_loss = sum(loss)/len(loss)
    total_loss.backward()
    if optimizer is None:
        pass
    else:
        optimizer.step()

    return total_loss.data

def mix_multi_task_train_epoch(csvfile, writer, model, Each_Attr_id, num_batches, train_loader, print_freq, optimizer=None):
    multi_label = MyBCEWithLogitsLoss()
    multi_class = torch.nn.CrossEntropyLoss()
    for batch_label in train_loader:
        loss = mix_multi_task_train_batch(model, Each_Attr_id, multi_label, multi_class, optimizer, batch_label)
        if num_batches%print_freq == 0:
            print('%23s%-9s%-13s'%('the '+str(num_batches)+'th batch, ','loss is: ',str(round(loss[0],8))))
        num_batches +=1
    return num_batches


########### get sample loss ###########

def mix_multi_task_test_loss_batch(txtFile, csvfile, writer, jpg_folders, model, Each_Attr_id, multi_label, multi_class, batch_label):
    multi_class_indexs = [0,1,2,6,8,9,12]
    multi_label_indexs = [3,4,5,7,10,11,13,14,15,16]
    num_each_attr = [5, 5, 5, 8, 5, 10, 10, 7, 12, 10, 7, 7, 4, 20, 23, 72, 47]

    C = batch_label[0][0].shape[0]
    H = batch_label[0][0].shape[1]
    W = batch_label[0][0].shape[2]
    batch = torch.cat((batch_label[i][0] for i in range(len(batch_label))),0).view(-1,C,H,W)  #  N C H W
    input = Variable(batch.cuda())

    labels = [batch_label[i][1] for i in range(len(batch_label))]  # [[1,-1..],[1,1,-1..]]
    for i, label in enumerate(labels):
        labels[i] = [1 if j==1 else 0 for j in label]
    index1_index0, branch_ids, _ = get_branch(Each_Attr_id, labels)

    output = model(input)  #  [NxC1, NxC2, NxC3, NxC4, NxC5, ..., NxC17]
    for ii,xx in enumerate(output):
        if ii==0:
            tensorOutput = xx.cpu().data
        else:
            tensorOutput = torch.cat((tensorOutput,xx.cpu().data),dim=1)
            
    sigmoid = torch.nn.functional.sigmoid
    tensorOutput = sigmoid(tensorOutput)
    npOutput = tensorOutput.numpy()
    
    
    loss = []
    sample_loss = [0.0]*len(batch_label)
    for i,(branch_id,label) in enumerate(zip(branch_ids, labels)):
        index0 = index1_index0[i]
        for index in branch_id:
            if index in multi_class_indexs:
                cur_output = torch.unsqueeze(output[index][index0], dim=0)
                if index==0:
                    start = 23
                else:
                    start = sum(num_each_attr[0:index])+23
                end = num_each_attr[index]+start
                for tt,qq in enumerate(label[start:end]):
                    if qq==1:
                        cur_label = Variable(torch.LongTensor([tt])).cuda()
                        sample_loss[i]+=float(multi_class(cur_output, cur_label).cpu().data)
                        loss.append(multi_class(cur_output, cur_label))
                        break
            elif index in multi_label_indexs:
                if index==0:
                    start = 23
                else:
                    start = sum(num_each_attr[0:index])+23
                end = num_each_attr[index]+start
                cur_output = torch.unsqueeze(output[index][index0], dim=0)
                cur_label  = Variable(torch.unsqueeze(torch.FloatTensor(label[start:end]), dim=0).cuda())
                sample_loss[i]+=float(multi_label(cur_output, cur_label).cpu().data)
                loss.append(multi_label(cur_output, cur_label))

    for test_loss, folder, confidences in zip(sample_loss, jpg_folders, npOutput):
        writer.writerow({'img path': folder, \
                         'total loss': round(test_loss,8)})
        txtFile.write(folder)
        for xx in confidences:
            txtFile.write(' '+str(round(xx,4)))
        txtFile.write('\n')



def mix_multi_task_test_loss_epoch(txtFile, csvfile, writer, model, Each_Attr_id, data_loader):
    multi_label = torch.nn.BCEWithLogitsLoss()
    multi_class = torch.nn.CrossEntropyLoss()
    for (batch_label,indexs) in tqdm(data_loader,desc='Test loss',ncols=100):
        jpg_folders = [data_loader.dataset.samples[index][0] for index in indexs]
        loss = mix_multi_task_test_loss_batch(txtFile, csvfile, writer, jpg_folders, model, Each_Attr_id, multi_label, multi_class, batch_label)
        

########### for sigmoid_cross_entropy_with_logits ##########


########### for softmax_cross_entropy_with_logits ##########
def train_batch(model, optimizer, batch_label):
    optimizer.zero_grad() #
    C = batch_label[0][0].shape[0]
    H = batch_label[0][0].shape[1]
    W = batch_label[0][0].shape[2]
    batch = torch.cat((batch_label[i][0] for i in range(len(batch_label))),0).view(-1,C,H,W)
    label = [batch_label[i][1] for i in range(len(batch_label))]
    input = Variable(batch.cuda())
    output = [i for i in model(input)]
    criterion = torch.nn.CrossEntropyLoss()
    loss = []
    true_label = GetTrueLabel(label)
    for i,j in enumerate(true_label):
        for Attr, Attr_Value in j:
            cur_output = torch.unsqueeze(output[Attr][i], dim=0)
            target = Variable(torch.LongTensor([Attr_Value]).cuda())
            loss.append(criterion(cur_output, target))
    Avg_loss = sum(loss)/input.shape[0]    # you should calcu each sample loss, and get a mean value
    Avg_loss.backward()

    optimizer.step()
    return Avg_loss.data

def train_epoch(model, num_batches, train_loader, print_freq, optimizer=None):
    for batch_label in train_loader:
        loss = train_batch(model, optimizer, batch_label)
        if num_batches%print_freq == 0:
            print('%23s%-9s%-13s'%('the '+str(num_batches)+'th batch, ','loss is: ',str(round(loss[0],8))))
        num_batches +=1
    return num_batches


########### for softmax_cross_entropy_with_logits ##########
#-*-coding:utf-8-*-
import os
import os.path as osp
import torch
from torch.autograd import Variable

def train_batch(model, optimizer, batch, label): 
    optimizer.zero_grad() # 
    input = Variable(batch)
    output = model(input)
    criterion = torch.nn.CrossEntropyLoss()
    criterion(output, Variable(label)).backward() 
    optimizer.step()
    return criterion(output, Variable(label)).data

def train_epoch(model, num_batches, train_loader, print_freq, optimizer=None):
    for batch, label in train_loader:
        loss = train_batch(model, optimizer, batch.cuda(), label.cuda())
        if num_batches%print_freq == 0:
            print('%23s%-9s%-13s'%('the '+str(num_batches)+'th batch, ','loss is: ',str(round(loss[0],8))))
        num_batches +=1
    return num_batches


def get_train_val_acc(model, train_loader, val_loader):
    model.eval()
    train_correct = 0
    train_total = 0

    val_correct = 0
    val_total = 0

    for i, (batch, label) in enumerate(val_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1]
        val_correct += pred_label.cpu().eq(label).sum()
        val_total += label.size(0)

    for i, (batch, label) in enumerate(train_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1]
        train_correct += pred_label.cpu().eq(label).sum()
        train_total += label.size(0)

    print("Train Accuracy :"+str(round( float(train_correct) / train_total , 3 )))
    print("Val   Accuracy :"+str(round( float(val_correct) / val_total , 3 )))

    model.train()
    return round( float(train_correct) / train_total , 3 ), round( float(val_correct) / val_total , 3 )

def get_train_val_test_acc(model, train_loader, val_loader, test_loader):
    model.eval()
    train_correct = 0
    train_total = 0

    val_correct = 0
    val_total = 0

    test_correct = 0
    test_total = 0

    for i, (batch, label) in enumerate(train_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1]
        train_correct += pred_label.cpu().eq(label).sum()
        train_total += label.size(0)

    for i, (batch, label) in enumerate(val_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1]
        val_correct += pred_label.cpu().eq(label).sum()
        val_total += label.size(0)

    for i, (batch, label) in enumerate(test_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1]
        test_correct += pred_label.cpu().eq(label).sum()
        test_total += label.size(0)

    print("Train Accuracy :"+str(round( float(train_correct) / train_total , 3 )))
    print("Val   Accuracy :"+str(round( float(val_correct) / val_total , 3 )))
    print("Test  Accuracy :"+str(round( float(test_correct) / test_total , 3 )))

    model.train()
    return round( float(train_correct) / train_total , 3 ), round( float(val_correct) / val_total , 3 ), round( float(test_correct) / test_total , 3 )

# do TenCrop operatios on test imags
def get_train_val_testTenCrop_acc(model, train_loader, val_loader, test_loader, TenCroptest_loader):
    model.eval()
    train_correct = 0
    train_total = 0

    val_correct = 0
    val_total = 0

    test_correct = 0
    test_total = 0

    TenCroptest_correct = 0
    TenCroptest_total = 0

    for i, (batch, label) in enumerate(train_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1]
        train_correct += pred_label.cpu().eq(label).sum()
        train_total += label.size(0)

    for i, (batch, label) in enumerate(val_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1]
        val_correct += pred_label.cpu().eq(label).sum()
        val_total += label.size(0)

    for i, (batch, label) in enumerate(test_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1]
        test_correct += pred_label.cpu().eq(label).sum()
        test_total += label.size(0)

    for i, (batch, label) in enumerate(TenCroptest_loader):
        bs, ncrops, c, h, w = batch.size()
        batch = batch.cuda()
        result = model(Variable(batch.view(-1, c, h, w))) # (10,100)
        result_avg = result.view(bs, ncrops, -1).mean(1)
        pred_label = result_avg.data.max(1)[1]
        TenCroptest_correct += pred_label.cpu().eq(label).sum()
        TenCroptest_total += label.size(0)


    print("Train         Accuracy :"+str(round( float(train_correct) / train_total , 3 )))
    print("Val           Accuracy :"+str(round( float(val_correct) / val_total , 3 )))
    print("Test          Accuracy :"+str(round( float(test_correct) / test_total , 3 )))
    print("TenCrop Test  Accuracy :"+str(round( float(TenCroptest_correct) / TenCroptest_total , 3 )))

    model.train()
    return round( float(val_correct) / val_total , 3 ), round( float(TenCroptest_correct) / TenCroptest_total , 3 )

def get_val_acc(model, val_loader):
    model.eval()
    val_correct = 0
    val_total = 0

    for i, (batch, label) in enumerate(val_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1]
        val_correct += pred_label.cpu().eq(label).sum()
        val_total += label.size(0)

    print("Val           Accuracy :"+str(round( float(val_correct) / val_total , 3 )))

    return round( float(val_correct) / val_total , 3 )



def get_val_test_acc(model, val_loader, test_loader):
    model.eval()
    val_correct = 0
    val_total = 0

    test_correct = 0
    test_total = 0

    for i, (batch, label) in enumerate(val_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1] 
        val_correct += pred_label.cpu().eq(label).sum() 
        val_total += label.size(0)

    for i, (batch, label) in enumerate(test_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1]
        test_correct += pred_label.cpu().eq(label).sum() 
        test_total += label.size(0)

    print("Val           Accuracy :"+str(round( float(val_correct) / val_total , 3 )))
    print("Test          Accuracy :"+str(round( float(test_correct) / test_total , 3 )))

    return round( float(val_correct) / val_total , 3 ), round( float(test_correct) / test_total , 3 )



def get_label(ind):
    if ind<=5:
        cur_label = Variable(torch.LongTensor([ind])).cuda()
        class_id = 0
    elif ind<=13:
        cur_label = Variable(torch.LongTensor([ind-6])).cuda()
        class_id = 1
    elif ind<=18:
        cur_label = Variable(torch.LongTensor([ind-14])).cuda()
        class_id = 2
    elif ind<=23:
        cur_label = Variable(torch.LongTensor([ind-19])).cuda()
        class_id = 3
    elif ind<=28:
        cur_label = Variable(torch.LongTensor([ind-24])).cuda()
        class_id = 4
    elif ind<=38:
        cur_label = Variable(torch.LongTensor([ind-29])).cuda()
        class_id = 5
    elif ind<=44:
        cur_label = Variable(torch.LongTensor([ind-39])).cuda()
        class_id = 6
    elif ind<=53:
        cur_label = Variable(torch.LongTensor([ind-45])).cuda()
        class_id = 7
    return class_id, cur_label


# batch=NxCxHxW
# label=Nxlength(such as length equals 54), one hot tensor
# true label shape should be N, value from 0 to C-1
def multi_task_train_batch(model, optimizer, batch, label): 
    optimizer.zero_grad() # 
    input = Variable(batch)
    output = model(input) # 8xNxC

    criterion = torch.nn.CrossEntropyLoss()

    loss = []
    for i,sample in enumerate(batch):
        sample = torch.unsqueeze(sample, dim=0)
        input = Variable(sample)
        ind = label[i]
        class_id, cur_label = get_label(ind)
        cur_output = output[class_id][i]
        cur_output = torch.unsqueeze(cur_output, dim=0)
        loss.append(criterion(cur_output, cur_label))
    total_loss = sum(loss)/len(loss)
    total_loss.backward() 
    optimizer.step()
    return total_loss.data


def multi_task_train_epoch(model, num_batches, train_loader, print_freq, optimizer=None):
    for batch, label in train_loader:
        loss = multi_task_train_batch(model, optimizer, batch.cuda(), label.cuda())
        if num_batches%print_freq == 0:
            print('%23s%-9s%-13s'%('the '+str(num_batches)+'th batch, ','loss is: ',str(round(loss[0],8))))
        num_batches +=1
    return num_batches

def multi_task_get_train_val_acc(model, train_loader, val_loader):
    model.eval()
    train_correct = 0
    train_total = 0

    val_correct = 0
    val_total = 0
    
    for i, (batch, label) in enumerate(train_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        for j,sample in enumerate(batch):
            sample = torch.unsqueeze(sample, dim=0)
            input = Variable(sample)
            ind = label[j]
            class_id, cur_label = get_label(ind)
            cur_label = cur_label.cpu().data
            cur_output = output[class_id][j]
            cur_output = torch.unsqueeze(cur_output, dim=0)
            pred_label = cur_output.data.max(1)[1]
            train_correct += pred_label.cpu().eq(cur_label).sum() # label为torch.LongTensor类型
            train_total += cur_label.size(0)
    print("multi_task Train Accuracy :"+str(round( float(train_correct) / train_total , 3 )))
    
    for i, (batch, label) in enumerate(val_loader):  #  batch_size = 1
        batch = batch.cuda()
        output = model(Variable(batch))
        ind = label[0]
        class_id, cur_label = get_label(ind)
        cur_output = output[class_id][0]
        cur_output = torch.unsqueeze(cur_output, dim=0)
        pred_label = cur_output.data.max(1)[1]
        cur_label = cur_label.cpu().data
        val_correct += pred_label.cpu().eq(cur_label).sum() # label为torch.LongTensor类型
        val_total += cur_label.size(0)
    print("multi_task Val   Accuracy :"+str(round( float(val_correct) / val_total , 3 )))

    model.train()
    return round( float(train_correct) / train_total , 3 ), round( float(val_correct) / val_total , 3 )

# undone
def multi_task_get_train_val_test_acc(model, train_loader, val_loader, test_loader):
    model.eval()
    train_correct = 0
    train_total = 0

    val_correct = 0
    val_total = 0

    test_correct = 0
    test_total = 0

    for i, (batch, label) in enumerate(train_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1]
        train_correct += pred_label.cpu().eq(label).sum()
        train_total += label.size(0)

    for i, (batch, label) in enumerate(val_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1]
        val_correct += pred_label.cpu().eq(label).sum()
        val_total += label.size(0)

    for i, (batch, label) in enumerate(test_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1]
        test_correct += pred_label.cpu().eq(label).sum()
        test_total += label.size(0)

    print("multi_task Train Accuracy :"+str(round( float(train_correct) / train_total , 3 )))
    print("multi_task Val   Accuracy :"+str(round( float(val_correct) / val_total , 3 )))
    print("multi_task Test  Accuracy :"+str(round( float(test_correct) / test_total , 3 )))

    model.train()
    return round( float(val_correct) / val_total , 3 ), round( float(test_correct) / test_total , 3 )
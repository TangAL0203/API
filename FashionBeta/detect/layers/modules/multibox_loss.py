# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from data import coco as cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1,0.2]

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        print(conf_t)

        pos = conf_t > 0
        #print(pos.size()) [32,8732]
        #print(conf_t.size()) [32,8732]
        num_pos = pos.sum(dim=1, keepdim=True)
        class_weight = torch.cuda.FloatTensor(
            [1.0, 2.925, 2.479, 1.448, 2.518, 1.719, 2.616, 3.526, 1.024, 1.832, 5.538])

        # 'hat', 'sunglass', 'upperclothes', 'skirt', 'pants',  'dress', 'belt', 'shoe', 'bag', 'scarf'
        pos_hat = conf_t == 1  # hat
        pos_sunglass = conf_t == 2  # sunglass
        pos_upperclothes = conf_t == 3  # upperclothes
        pos_skirt = conf_t == 4  # skirt
        pos_pants = conf_t == 5  # pants
        pos_dress = conf_t == 6
        pos_belt = conf_t == 7
        pos_shoe = conf_t == 8
        pos_bag = conf_t == 9
        pos_scarf = conf_t == 10
        # hat
        pos_hat_idx = pos_hat.unsqueeze(pos_hat.dim()).expand_as(loc_data)
        loc_p_hat = loc_data[pos_hat_idx].view(-1, 4)
        loc_t_hat = loc_t[pos_hat_idx].view(-1, 4)
        loss_l_hat = 2.925 * F.smooth_l1_loss(loc_p_hat, loc_t_hat, size_average=False)
        print('hat:',loss_l_hat)
        # sunglass
        pos_sunglass_idx = pos_sunglass.unsqueeze(pos_sunglass.dim()).expand_as(loc_data)
        loc_p_sunglass = loc_data[pos_sunglass_idx].view(-1, 4)
        loc_t_sunglass = loc_t[pos_sunglass_idx].view(-1, 4)
        loss_l_sunglass = 2.479 * F.smooth_l1_loss(loc_p_sunglass, loc_t_sunglass, size_average=False)
        print('sunglass:',loss_l_sunglass)
        # upper
        pos_upperclothes_idx = pos_upperclothes.unsqueeze(pos_upperclothes.dim()).expand_as(loc_data)
        loc_p_up = loc_data[pos_upperclothes_idx].view(-1, 4)
        loc_t_up = loc_t[pos_upperclothes_idx].view(-1, 4)
        loss_l_upperclothes = 1.448 * F.smooth_l1_loss(loc_p_up, loc_t_up, size_average=False)
        # skirt
        pos_skirt_idx = pos_skirt.unsqueeze(pos_skirt.dim()).expand_as(loc_data)
        loc_p_skirt = loc_data[pos_skirt_idx].view(-1, 4)
        loc_t_skirt = loc_t[pos_skirt_idx].view(-1, 4)
        loss_l_skirt = 2.518 * F.smooth_l1_loss(loc_p_skirt, loc_t_skirt, size_average=False)
        # pants
        pos_pants_idx = pos_pants.unsqueeze(pos_pants.dim()).expand_as(loc_data)
        loc_p_pants = loc_data[pos_pants_idx].view(-1, 4)
        loc_t_pants = loc_t[pos_pants_idx].view(-1, 4)
        loss_l_pants = 1.719 * F.smooth_l1_loss(loc_p_pants, loc_t_pants, size_average=False)
        # dress
        pos_dress_idx = pos_dress.unsqueeze(pos_dress.dim()).expand_as(loc_data)
        loc_p_dress = loc_data[pos_dress_idx].view(-1, 4)
        loc_t_dress = loc_t[pos_dress_idx].view(-1, 4)
        loss_l_dress = 2.616 * F.smooth_l1_loss(loc_p_dress, loc_t_dress, size_average=False)
        print('dress:',loss_l_dress)
        # belt
        pos_belt_idx = pos_belt.unsqueeze(pos_belt.dim()).expand_as(loc_data)
        loc_p_belt = loc_data[pos_belt_idx].view(-1, 4)
        loc_t_belt = loc_t[pos_belt_idx].view(-1, 4)
        loss_l_belt = 3.526 * F.smooth_l1_loss(loc_p_belt, loc_t_belt, size_average=False)
        # shoe
        pos_shoe_idx = pos_shoe.unsqueeze(pos_shoe.dim()).expand_as(loc_data)
        loc_p_shoe = loc_data[pos_shoe_idx].view(-1, 4)
        loc_t_shoe = loc_t[pos_shoe_idx].view(-1, 4)
        loss_l_shoe = 1.024 * F.smooth_l1_loss(loc_p_shoe, loc_t_shoe, size_average=False)
        # bag
        pos_bag_idx = pos_bag.unsqueeze(pos_bag.dim()).expand_as(loc_data)
        loc_p_bag = loc_data[pos_bag_idx].view(-1, 4)
        loc_t_bag = loc_t[pos_bag_idx].view(-1, 4)
        loss_l_bag = 1.832 * F.smooth_l1_loss(loc_p_bag, loc_t_bag, size_average=False)
        # scarf
        pos_scarf_idx = pos_scarf.unsqueeze(pos_scarf.dim()).expand_as(loc_data)
        loc_p_scarf = loc_data[pos_scarf_idx].view(-1, 4)
        loc_t_scarf = loc_t[pos_scarf_idx].view(-1, 4)
        loss_l_scarf = 5.538 * F.smooth_l1_loss(loc_p_scarf, loc_t_scarf, size_average=False)

        loss_l = loss_l_hat + loss_l_sunglass + loss_l_upperclothes + loss_l_skirt + loss_l_pants\
                 + loss_l_dress + loss_l_belt + loss_l_shoe + loss_l_bag + loss_l_scarf
        '''
        # Localization Loss (Smooth L1)
        # loc data Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        print('loss_l:',loss_l)
        print(loss_l.size())
        '''

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        #print(loss_c.size())  #[279424, 1]
        #print(pos.size()) #[32, 8732]

        # Hard Negative Mining
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)  #[32,8732]
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        #print(neg)
        #print(neg.size())#[32,8732]

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data) #[32,8732,11]
        #print(pos_idx)  #[32x8732x11]
        neg_idx = neg.unsqueeze(2).expand_as(conf_data) #[32,8732,11]
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        #print(conf_p.size())
        targets_weighted = conf_t[(pos+neg).gt(0)]
        #print(targets_weighted.size())
        #class_weight = torch.cuda.FloatTensor([1.0, 8.556, 6.146, 2.097, 6.339, 2.954, 6.846, 12.432, 1.049, 3.358, 30.675])
        #class_weight = torch.cuda.FloatTensor([1.0,2.925, 2.479, 1.448, 2.518, 1.719, 2.616, 3.526, 1.024, 1.832, 5.538])
        loss_c = F.cross_entropy(conf_p, targets_weighted,weight=class_weight,size_average=False)
        print('loss_c:',loss_c)
        print(loss_c.size())

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c

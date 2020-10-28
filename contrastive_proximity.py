import torch
import torch.nn as nn

class Con_Proximity(nn.Module):

    def __init__(self, num_classes=100, feat_dim=1024, use_gpu=True):
        super(Con_Proximity, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())#100 x feats- for 100 centers
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)

        '''
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
        torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        '''


        distmat = torch.norm(x[:, None, :] - self.centers[None, :, :], dim = 2) ** 2
        classes = torch.arange(self.num_classes).long()


        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            k = mask[i].clone().to(dtype=torch.long)
            k = 1-k
            kk = k.clone().to(dtype=torch.long)

            value = distmat[i][kk] # 找出非对应类别的距离
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss

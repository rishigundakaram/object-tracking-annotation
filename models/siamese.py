import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.optim import Adam 

from statistics import mean
from random import choices, choice, seed
import matplotlib.pyplot as plt
import numpy as np

class image_encoder(nn.Module): 
    def __init__(self, size=32) -> None:
        super(image_encoder, self).__init__()
        self.size = size
        self.encoder = nn.Sequential()
        
    def rescale(self, image): 
        width, height, _ = np.shape(image)
        num_w_boxes = width // self.size
        num_h_boxes = height // self.size
        max_w_size = self.size * num_w_boxes
        max_h_size = self.size * num_h_boxes
        return num_w_boxes, num_w_boxes, image.thumbnail((max_w_size, max_h_size))
    
    def forward(self, image): 
        num_w_boxes, num_h_boxes, image = self.rescale(image)
        return num_w_boxes, num_h_boxes, self.encoder(image)

class object_encoder(nn.Module): 
    def __init__(self, size=32) -> None:
        super(object_encoder, self).__init__()
        self.size = size
        self.encoder = nn.Sequential()
        
    def rescale(self, image): 
        return image.thumbnail((self.size, self.size))
    
    def forward(self, image, rescale=True): 
        if rescale: 
            image = self.rescale(image)
        return self.encoder(image)

class SiameseDetector(nn.Module): 
    def __init__(self, backbone=None, size=32, preds=1): 
        super(SiameseDetector, self).__init__()
        self.preds = preds
        self.image_encoder = image_encoder(size=size)
        self.object_encoder = object_encoder(size=size)
        self.features = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
        )
        # represents either same or not and confidence
        self.sim_conf = nn.Sequential(
            nn.Linear(64, 1 + self.preds),
            nn.Sigmoid(),
        )
        # each possible bounding box
        self.bb_regression = nn.Sequential(
            nn.Linear(64, 4*self.preds),
        )
        
    def forward(self, img, obj): 
        num_w_boxes, num_h_boxes, img = self.image_encoder(img)
        obj = self.object_encoder(obj)
        images = torch.split(img, [num_w_boxes, num_h_boxes])
        output = torch.zeros((num_w_boxes, num_h_boxes, 5 * self.preds))
        idx_w = 0
        idx_h = 0
        for img in enumerate(images): 
            if idx_w == num_w_boxes:
                idx_w = 0
                idx_h += 1
            feature = self.features(torch.abs(img - obj))
            sim = self.sim(feature)
            bb_regression = self.bb_regression(feature)
            out = torch.cat((sim, bb_regression))
            output[idx_w, idx_h, :] = out
        return output

class Triplet_Dataset(torch.utils.data.Dataset): 
    def __init__(self, dataset):
        self.data = dataset
        self.len = len(dataset) ** 2
    
    def __getitem__(self,idx): 
        i = idx % len(self.data)
        j = idx // len(self.data)
        label = 0
        if self.data[i][1] == self.data[j][1]: 
            label = 1
        return (self.data[i][0], self.data[j][0], torch.tensor([label]))
    def __len__(self): 
        return self.len

def eval(loader, model):
    same_score = []
    diff_score = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad(): 
        for anchor, pos, neg in loader: 
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            dist_pos = model(anchor, pos)
            dist_neg = model(anchor, neg)

            same = dist_pos.reshape(-1).tolist()
            same_score.extend(same)

            diff = dist_neg.reshape(-1).tolist()
            diff_score.extend(diff)
    return mean(same_score), mean(diff_score)

class SiameseNN(nn.Module): 
    def __init__(self, backbone=None, distance_dim=4096): 
        super(SiameseNN, self).__init__()
        self.distance_dim = distance_dim
        if backbone is None: 
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        else: 
            self.backbone = backbone
        self.sim = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, base_im, cmp_im): 
        out_1 = self.backbone(base_im)
        out_2 = self.backbone(cmp_im)
        return self.sim(torch.abs(out_1 - out_2))
    
    
class Triplet_Dataset(torch.utils.data.Dataset): 
    def __init__(self, train_classes, num_samples, seed, train=True) -> None:
        super().__init__()
        
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.fullset = torchvision.datasets.CIFAR100(root='../data/processed', train=True,
                                                    download=True, transform=transform)
        
        self.train_classes = train_classes
        self.num_samples = num_samples
        self.seed = seed
        
        self._create_dataset(train=train)

    def __len__(self): 
        return len(self.anchors)
    
    def __getitem__(self, idx): 
        return self.anchors[idx, ...], self.im_positive[idx, ...], self.im_negative[idx, ...]
    
    def create_label_map(self): 
        seed(self.seed)
        labels = {}
        for i in range(len(self.fullset)): 
            if self.fullset[i][1] in labels: 
                labels[self.fullset[i][1]].append(i)
            else: 
                labels[self.fullset[i][1]] = [i]
        return labels

    def train_test_split(self): 
        train_classes = choices(list(range(100)), k=self.train_classes)
        valid_classes = [i for i in range(100) if i not in train_classes]
        return train_classes, valid_classes

    def _create_dataset(self, train=True): 
        anchors = torch.zeros((self.num_samples, 3, 32, 32), dtype=torch.float)
        im_positive = torch.zeros((self.num_samples, 3, 32, 32), dtype=torch.float)
        im_negative = torch.zeros((self.num_samples, 3, 32, 32), dtype=torch.float)
        train_classes, valid_classes = self.train_test_split()
        label_map = self.create_label_map()
        potential_classes = train_classes
        if not train: 
            potential_classes = valid_classes
        for idx in range(self.num_samples): 
            true_class = choice(potential_classes)
            false_class = choice(potential_classes)
            while false_class == true_class: 
                false_class = choice(potential_classes)
            anchor_idx = choice(label_map[true_class])
            pos_idx = choice(label_map[true_class])
            neg_idx = choice(label_map[false_class])
            anchors[idx, :, :, :] = self.fullset[anchor_idx][0]
            im_positive[idx, :, :, :] = self.fullset[pos_idx][0]
            im_negative[idx, :, :, :] = self.fullset[neg_idx][0]
        self.anchors, self.im_positive, self.im_negative = anchors, im_positive, im_negative


    
class Triplet_Loss: 
    def __init__(self, margin) -> None:
        self.margin = margin

    def __call__(self, dist_same, dist_diff):
        return torch.mean(torch.clamp(dist_same - dist_diff + self.margin,min=0.0))
        


if __name__ == '__main__' and '__file__' in globals(): 
    print(f'using gpu: {torch.cuda.is_available()}')
    config = {
        'lr': .0001, 
        'batchsize': 32, 
        'betas': (.9, .99), 
        'distance_dim': 4096, 
        'epochs': 350,
        'margin': .9,
        'train_classes': 75, 
        'train_samples': 50_000, 
        'val_samples': 10_000
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNN(distance_dim=config['distance_dim']).to(device)
    optimizer = Adam(params=model.parameters(), lr=config['lr'], betas=config['betas'])
    print('optimizer + model loaded')
    
    
    trainset = Triplet_Dataset(config['train_classes'], config['train_samples'], 0, train=True)
    valset = Triplet_Dataset(config['train_classes'], config['val_samples'], 0, train=False)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config['batchsize'],
                                          shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=config['batchsize'],
                                          shuffle=True, num_workers=8)

    print(f'number of samples: {len(trainset)}')
    loss = Triplet_Loss(margin=config['margin'])
    ep = config['epochs']
    
    
    model.train()
    for e in range(ep): 
        errs = []
        same_score = []
        diff_score = []
        acc = 0
        model.train()
        for anchor, pos, neg  in train_loader: 
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            dist_pos = model(anchor, pos)
            dist_neg = model(anchor, neg)
            
            err = loss(dist_pos, dist_neg)

            same = dist_pos.reshape(-1).tolist()
            same_score.extend(same)

            diff = dist_neg.reshape(-1).tolist()
            diff_score.extend(diff)

            err.backward()
            optimizer.step()
            optimizer.zero_grad()
            errs.append(err.item())
        val_same_mean, val_diff_mean = eval(val_loader, model)
        print(f'epoch: {e}, train_error: {mean(errs)}, train_sim_score: {round(mean(same_score), 3)}, train_diff_score: {round(mean(diff_score), 3)}, val_sim_score: {round(val_same_mean, 3)}, val_diff_score: {round(val_diff_mean, 3)}')
import torch
import torchvision
import torchvision.transforms as transforms


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

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset_1 = torchvision.datasets.CIFAR10(root='../data/processed', train=True,
                                                download=True, transform=transform)
    trainset_2 = torchvision.datasets.CIFAR10(root='../data/processed', train=True,
                                                download=True, transform=transform)
                                                
    trainloader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=1,
                                        shuffle=True, num_workers=5)
    trainloader_2 = torch.utils.data.DataLoader(trainset_2, batch_size=1,
                                        shuffle=True, num_workers=5)    

    
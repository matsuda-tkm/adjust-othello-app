import torch.nn as nn

class DuelingNet(nn.Module):
    def __init__(self, c=100):
        super().__init__()
        self.c = c
        self.conv_layer = nn.Sequential(
            nn.Conv2d(2, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU()
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(c*64, 256),
            nn.ReLU()
        )
        self.adv_layer = nn.Linear(256, 65)
        self.val_layer = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, self.c*64)
        x = self.fc_layer(x)
        adv = self.adv_layer(x)
        val = self.val_layer(x).expand(-1,65)
        out = adv + val - adv.mean(1, keepdim=True).expand(-1,65)
        return out.tanh()

class ValueNet(nn.Module):
    def __init__(self, c=100):
        super().__init__()
        self.c = c
        self.conv_layer = nn.Sequential(
            nn.Conv2d(2, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU()
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(c*64, 256),
            nn.ReLU()
        )
        self.adv_layer = nn.Linear(256, 65)
        self.val_layer = nn.Linear(256, 1)
        self.output_layer = nn.Linear(65, 1)
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, self.c*64)
        x = self.fc_layer(x)
        adv = self.adv_layer(x)
        val = self.val_layer(x).expand(-1,65)
        out = adv + val - adv.mean(1, keepdim=True).expand(-1,65)
        out = self.output_layer(out)
        return out
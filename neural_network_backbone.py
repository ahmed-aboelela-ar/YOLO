import torch 
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    

class TinyYoloV1(nn.Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=2):
        super(TinyYoloV1, self).__init__()
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        
        self.output_dim = self.S * self.S * (self.B * 5 + self.C)

        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 16, kernel_size=7, stride=2, padding=3), #O1
            nn.MaxPool2d(kernel_size=2, stride=2), #O2
            ConvBlock(16, 48, kernel_size=3, stride=1, padding=1), #O3
            nn.MaxPool2d(kernel_size=2, stride=2), #O4
            ConvBlock(48, 32, kernel_size=1, stride=1, padding=0), #O5
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1), #O6
            ConvBlock(64, 64, kernel_size=1, stride=1, padding=0), #O7
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1), #O8
            nn.MaxPool2d(kernel_size=2, stride=2),  #O9
            ConvBlock(128, 64, kernel_size=1, stride=1, padding=0), #10
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1), #11
            ConvBlock(128, 64, kernel_size=1, stride=1, padding=0), #12
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1), #13
            ConvBlock(128, 64, kernel_size=1, stride=1, padding=0), #14
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1), #15
            ConvBlock(128, 64, kernel_size=1, stride=1, padding=0), #16
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1), #17
            ConvBlock(128, 128, kernel_size=1, stride=1, padding=0), #18
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1), #19
            nn.MaxPool2d(kernel_size=2, stride=2), #20
            ConvBlock(256, 128, kernel_size=1, stride=1, padding=0), #21
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1), #22
            ConvBlock(256, 128, kernel_size=1, stride=1, padding=0), #23
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1), #24
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1), #25
            ConvBlock(256, 256, kernel_size=3, stride=2, padding=1), #26
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1), #27
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1), #28
        )
        
        
        # nn.Sequential(
        #     ConvBlock(in_channels, 16, kernel_size=3, stride=2, padding=1),
            
        #     ConvBlock(16, 32, kernel_size=3, stride=2, padding=1),
            
        #     ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
            
        #     ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            
        #     ConvBlock(128, 128, kernel_size=3, stride=2, padding=1),
            
        #     ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
        # )
        
        
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512), 
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(512, self.output_dim) 
        )
        
        # nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(1024 * 7 * 7, 496),
        #     nn.Dropout(0.5), # Standard is 0.5, you had 0.0
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(496, self.output_dim) # Now Dynamic (588)
        # )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        
        # Reshape to (Batch, 7, 7, 12)
        x = x.view(-1, self.S, self.S, (self.B * 5 + self.C))
        
        return x
    

if __name__ == '__main__':
    model = TinyYoloV1()

    img = torch.zeros((1, 3, 448, 448))
    out = model(img)
    
    print(out.shape)
    
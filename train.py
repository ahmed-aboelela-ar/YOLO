from neural_network_backbone import TinyYoloV1
from synthetic_voc_dataset import VocDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from loss import YoloLoss


if __name__ == '__main__':
    model = TinyYoloV1()
    
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize((448, 448)),
        # transforms.ToTensor()
    ])

    dataset_train = VocDataset('synthetic_voc', split='train', transform=transform)
    dataset_val = VocDataset('synthetic_voc', split='val', transform=transform)


    train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(dataset_val, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)


    loss = YoloLoss(S=7, B=2, C=2)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    model.to(device)
    loss.to(device)


    EPOCHS = 100
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        loop_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_dataloader):
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)
            
            loss_value = loss(predictions, targets)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            loop_loss += loss_value.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(train_dataloader)}] Loss: {loss_value.item():.4f}")

        mean_loss = loop_loss / len(train_dataloader)
        print(f"=== Epoch [{epoch+1}/{EPOCHS}] Completed. Average Loss: {mean_loss:.4f} ===")

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_idx, (images, targets) in enumerate(val_dataloader): 
                images = images.to(device)
                targets = targets.to(device)

                predictions = model(images)
                val_loss += loss(predictions, targets).item()
            val_loss /= len(val_dataloader)
            print(f"Validation Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "tiny_yolo_v1_best.pth")
                print("Best model saved!")



    torch.save(model.state_dict(), "tiny_yolo_v1.pth")
    print("Training finished and model saved to 'tiny_yolo_v1.pth'")

    

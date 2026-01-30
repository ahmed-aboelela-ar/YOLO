from neural_network_backbone import TinyYoloV1
from synthetic_voc_dataset import VocDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import os
from loss import YoloLoss
from map import mAP
import albumentations  as  A
from albumentations.pytorch import ToTensorV2
import time
from pathlib import Path
import datetime
from torch.utils.tensorboard import SummaryWriter
import torchvision
import cv2
from utils import draw_and_interpret

if __name__ == '__main__':
    cv2.setNumThreads(0)
    model = TinyYoloV1()
    log_dir = Path(f"logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    train_transform = A.Compose([
        A.RandomCrop(width=448, height=448),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Blur(p=0.3),
        A.GaussNoise(p=0.1, std_range=(0.1, 0.5)),
        A.RandomBrightnessContrast(p=0.2),

    ],
    bbox_params=A.BboxParams(
        format='yolo',

        label_fields=['class_labels']
    )
    )

    dataset_train = VocDataset('synthetic_voc', split='train', transform=train_transform)
    dataset_val = VocDataset('synthetic_voc', split='val')

    


    train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(dataset_val, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)


    loss = YoloLoss(S=7, B=2, C=2)
    EPOCHS = 80
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    model.to(device)
    loss.to(device)

    map_metric = mAP(device)

    # if os.path.exists("tiny_yolo_v1_best.pth"):
    #     model.load_state_dict(torch.load("tiny_yolo_v1_best.pth"))
    #     print("Model loaded from 'tiny_yolo_v1_best.pth'")

    model_saving_path =  Path('models')
    now = datetime.datetime.now()
    current_models_saving_path = model_saving_path / now.strftime("%Y-%m-%d_%H-%M-%S")
    current_models_saving_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    fixed_val_images, fixed_val_targets = next(iter(val_dataloader))
    fixed_val_images = fixed_val_images.to(device)
    

    dataiter = iter(train_dataloader)
    images, labels = next(dataiter)
    images = images.to(device)
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image('images', img_grid)
    writer.add_graph(model, images)
    



    for epoch in range(EPOCHS):
        model.train()
        loop_loss = 0
        writer.add_scalar('Train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
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
        writer.add_scalar('Loss/train', mean_loss, epoch)
        print(f"=== Epoch [{epoch+1}/{EPOCHS}] Completed. Average Loss: {mean_loss:.4f} ===")

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_idx, (images, targets) in enumerate(val_dataloader): 
                images = images.to(device)
                targets = targets.to(device)

                predictions = model(images)
                if batch_idx == 0:
                    img_np = images[:8].detach().cpu().numpy()
                    pred_np = predictions[:8].detach().cpu().numpy()
                    for i in range(8):
                        pred_img = draw_and_interpret(img_np[i], pred_np[i])
                        writer.add_image(f'Val_Images/{i}', torch.tensor(pred_img).permute(2, 0, 1), epoch)

                val_loss += loss(predictions, targets).item()
                map_metric.update(predictions, targets)
            val_loss /= len(val_dataloader)
            writer.add_scalar('Loss/val', val_loss, epoch)
            print(f"Validation Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), current_models_saving_path / f"tiny_yolo_v1_{epoch+1}.pth")
                print("Best model saved!")

        if (epoch + 1) % 10 == 0:
            print("Calculating mAP...")
            map50, map_all = map_metric.compute()
            writer.add_scalar('mAP/mAP@0.5', map50, epoch)
            writer.add_scalar('mAP/mAP@0.5:0.95', map_all, epoch)
            print(f"--> mAP@0.5: {map50:.4f} | mAP@0.5:0.95: {map_all:.4f}")
        else:
            map_metric.metric.reset()
        
        scheduler.step()



    torch.save(model.state_dict(), current_models_saving_path / f"tiny_yolo_v1_{epoch+1}.pth")
    print(f"Model saved to {current_models_saving_path / f'tiny_yolo_v1_{epoch+1}.pth'}")

    

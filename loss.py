import torch
import torch.nn as nn
from utils import get_analysis_masks

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        self.S = S
        self.B = B
        self.C = C
        
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + 5 * self.B)
        
        best_box_idx, no_obj_mask = get_analysis_masks(predictions, target, self.B)

        exists_box = target[..., 4].unsqueeze(-1) 

        pred_boxes = predictions[..., :self.B*5].reshape(-1, self.S, self.S, self.B, 5)
        
        gather_idx = best_box_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 5)
        best_box_pred = torch.gather(pred_boxes, dim=3, index=gather_idx).squeeze(3)

        box_loss_xy = self.mse(
            best_box_pred[..., 0:2] * exists_box, 
            target[..., 0:2] * exists_box
        )
        
        box_loss_wh = self.mse(
            torch.sqrt(torch.abs(best_box_pred[..., 2:4]) + 1e-6) * exists_box, 
            torch.sqrt(target[..., 2:4]) * exists_box
        )
        
        object_loss = self.mse(
            best_box_pred[..., 4:5] * exists_box,
            target[..., 4:5] * exists_box
        )
        
        no_object_loss = self.mse(
            predictions * no_obj_mask,
            target * no_obj_mask
        )

        class_start_idx = 5 * self.B
        class_loss = self.mse(
            predictions[..., class_start_idx:] * exists_box,
            target[..., class_start_idx:] * exists_box
        )

        loss = (
            self.lambda_coord * box_loss_xy +
            self.lambda_coord * box_loss_wh +
            object_loss +
            self.lambda_noobj * no_object_loss +
            class_loss
        )

        return loss


        




if __name__ == "__main__":
    loss = YoloLoss(S=1, B=2, C=3)
    predictions = torch.zeros((1, 1, 1, 13))
    target = torch.zeros((1, 1, 1, 13))


    predictions[0, 0, 0, :] = torch.tensor([
        0.2, 0.7, 0.64, 0.25, 0.6,
        0.9, 0.9, 0.1, 0.1, 0.8,
        0.2, 0.7, 0.1
    ]) 

    target[0, 0, 0, :] = torch.tensor([
        0.2, 0.8, 0.64, 0.36, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0
    ])
    

        
        
        
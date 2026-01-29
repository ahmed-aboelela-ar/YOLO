from constants import CLASS_MAPPING, MAP_IOU_THRESHOLDS, S, B, C
import torch
from utils import from_yolo_to_actual_coord, calculate_IOU, get_confidences, cellboxes_to_boxes
from torchmetrics.detection import MeanAveragePrecision
import torch.nn as nn


class mAP(nn.Module):
    def __init__(self, threshold=0.5, device='cpu'):
        super().__init__()
        self.threshold = threshold
        self.metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
        self.metric.to(device)
        self.device = device



    def update(self, predictions, targets):
        pred_cellboxes = cellboxes_to_boxes(predictions)
        target_cellboxes = cellboxes_to_boxes(targets)

        pred_list = []
        target_list = []

        pred_mask = pred_cellboxes[..., 4] > 0.1
        target_mask = target_cellboxes[..., 4] > 0.5

        pred_count = pred_mask.sum(dim=1).cpu().tolist()
        target_count = target_mask.sum(dim=1).cpu().tolist()

        all_valid_pred = pred_cellboxes[pred_mask]
        all_valid_target = target_cellboxes[target_mask]

        pred_splits = torch.split(all_valid_pred, pred_count)
        target_splits = torch.split(all_valid_target, target_count)

        pred_list = [
            {
                "boxes": p[:, 0:4],
                "scores": p[:, 4],
                "labels": p[:, 5].long()
            } for p in pred_splits
        ]
        
        target_list = [
            {
                "boxes": t[:, 0:4],
                "labels": t[:, 5].long()
            } for t in target_splits
        ]
        
        self.metric.update(pred_list, target_list)
        

        

    def compute(self):
        result = self.metric.compute()
        self.metric.reset()
        return result['map_50'], result['map']



                



if __name__ == "__main__":
    predictions = torch.zeros((1, 1, 1, 12))
    predictions[0, 0, 0, 0:5] = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.9])
    predictions[0, 0, 0, 5:10] = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.8]) 
    predictions[0, 0, 0, 10]   = 0.7


    
    targets = torch.zeros((1, 1, 1, 12))
    targets[0, 0, 0, 0:5] = torch.tensor([0.5, 0.5, 0.5, 0.5, 1.0])
    targets[0, 0, 0, 10] = 1.0

    map_metric = mAP()
    map_metric.mAP(predictions, targets)
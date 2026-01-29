import torch 
import torch.nn as nn
import cv2
from constants import CLASSES, CONF_THRESHOLD, S
import numpy as np
from torchvision.ops import nms


def get_corners(xy, wh):
    half_wh = wh / 2
    return torch.cat([xy - half_wh, xy + half_wh], dim=-1)

def from_yolo_to_actual_coord(grid):
    device = grid.device
    B, S, _, _ = grid.shape

    idx = torch.arange(S, device=device) 
    row_idx, col_idx = torch.meshgrid(idx, idx, indexing='ij')
    
    xy_grid = torch.stack((col_idx, row_idx), dim=-1).unsqueeze(0)
    
    box1_xy = grid[..., 0:2]
    box1_wh = grid[..., 2:4]
    box2_xy = grid[..., 6:8]
    box2_wh = grid[..., 8:10]

    box1_xy_global = (box1_xy + xy_grid) / S
    box2_xy_global = (box2_xy + xy_grid) / S

    return get_corners(box1_xy_global, box1_wh), get_corners(box2_xy_global, box2_wh)

def get_confidences(grid):
    conf1 = grid[..., 4]
    conf2 = grid[..., 9]
    return conf1, conf2


def cellboxes_to_boxes(predictions):
    batch_size = predictions.shape[0]

    corners1, corners2 = from_yolo_to_actual_coord(predictions)
    conf1, conf2 = get_confidences(predictions)

    corners1 = corners1.reshape(batch_size, -1, 4)
    corners2 = corners2.reshape(batch_size, -1, 4)

    conf1 = conf1.reshape(batch_size, -1, 1)
    conf2 = conf2.reshape(batch_size, -1, 1)

    class_probs = predictions[..., 10:]
    best_class = torch.argmax(class_probs, dim=-1).unsqueeze(-1).reshape(batch_size, -1, 1)
    
    box1_vec = torch.cat([corners1, conf1, best_class], dim=-1)
    box2_vec = torch.cat([corners2, conf2, best_class], dim=-1)
    
    all_boxes = torch.cat([box1_vec, box2_vec], dim=1)
    return all_boxes
     



def calculate_IOU(pred_boxes_actual, target_boxes):
    inter_box = torch.zeros_like(pred_boxes_actual)
    
    inter_box[..., 0] = torch.max(pred_boxes_actual[..., 0], target_boxes[..., 0])
    inter_box[..., 1] = torch.max(pred_boxes_actual[..., 1], target_boxes[..., 1])
    inter_box[..., 2] = torch.min(pred_boxes_actual[..., 2], target_boxes[..., 2])
    inter_box[..., 3] = torch.min(pred_boxes_actual[..., 3], target_boxes[..., 3])
   
    inter_area = torch.clamp(inter_box[..., 2] - inter_box[..., 0], min=0) * torch.clamp(inter_box[..., 3] - inter_box[..., 1], min=0)
    
    pred_area = (pred_boxes_actual[..., 2] - pred_boxes_actual[..., 0]) * (pred_boxes_actual[..., 3] - pred_boxes_actual[..., 1])
    target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
    
    union_area = pred_area + target_area - inter_area
    
    return inter_area / (union_area+1e-6)


def get_analysis_masks(pred_grid, target_grid, B):
    pred_corners1, pred_corners2 = from_yolo_to_actual_coord(pred_grid)
    target_corners, _ = from_yolo_to_actual_coord(target_grid)
    
    iou1 = calculate_IOU(pred_corners1, target_corners).unsqueeze(-1)
    iou2 = calculate_IOU(pred_corners2, target_corners).unsqueeze(-1)
    
    ious = torch.cat([iou1, iou2], dim=-1)
    
    best_box_idx = torch.argmax(ious, dim=-1)
    
    no_obj_mask = torch.ones_like(target_grid)
    with_obj_mask = target_grid[..., 4] == 1

    best_box_conf_idx = (best_box_idx * 5) + 4

    
    no_obj_mask[with_obj_mask] = no_obj_mask[with_obj_mask].scatter_(
        dim=-1, 
        index=best_box_conf_idx[with_obj_mask].unsqueeze(-1).long(), 
        value=0
    )
    
    
    return best_box_idx, no_obj_mask

    

def draw_and_interpret(img, pred_grid, target_grid=None):
    
    img = np.ascontiguousarray(img)
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    img = np.ascontiguousarray(img)
    img_h, img_w = img.shape[0], img.shape[1]

    if pred_grid.ndim == 4:
        pred_grid = pred_grid[0]
        
    S = pred_grid.shape[0]
    

    boxes_list = []
    scores_list = []
    class_indices_list = []

    for i in range(S):
        for j in range(S):
            conf1 = pred_grid[i, j, 4].item()
            conf2 = pred_grid[i, j, 9].item()

            if conf1 > CONF_THRESHOLD or conf2 > CONF_THRESHOLD:
                if conf1 >= conf2:
                    box_data = pred_grid[i, j, 0:4]
                    conf = conf1
                else:
                    box_data = pred_grid[i, j, 5:9]
                    conf = conf2
                
                x_global = (j + box_data[0].item()) / S
                y_global = (i + box_data[1].item()) / S
                w_global = abs(box_data[2].item())
                h_global = abs(box_data[3].item())
                
                x_c = x_global * img_w
                y_c = y_global * img_h
                w_pix = w_global * img_w
                h_pix = h_global * img_h
                
                x1 = x_c - w_pix / 2
                y1 = y_c - h_pix / 2
                x2 = x_c + w_pix / 2
                y2 = y_c + h_pix / 2
                
                class_probs = pred_grid[i, j, 10:]
                class_idx = np.argmax(class_probs)

                boxes_list.append([x1, y1, x2, y2])
                scores_list.append(conf)
                class_indices_list.append(class_idx)

    if boxes_list:
        boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
        scores_tensor = torch.tensor(scores_list, dtype=torch.float32)
        
        keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)

        for idx in keep_indices:
            idx = idx.item()
            x1, y1, x2, y2 = boxes_list[idx]
            conf = scores_list[idx]
            class_idx = class_indices_list[idx]
            class_name = CLASSES[class_idx]

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(img_w, int(x2))
            y2 = min(img_h, int(y2))

            color = (0, 255, 0) if class_idx == 0 else (255, 0, 0)
            
            disp_conf = min(conf, 1.0)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{class_name} {disp_conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img
    

    



if __name__ == "__main__":
    
    test_tensor = torch.rand((8, 7, 7, 12))
    cellboxes_to_boxes(test_tensor)
    
    
    
    # pred_grid = torch.zeros((1, 2, 2, 12))
    # target_grid = torch.zeros((1, 2, 2, 12))

    # pred_grid[0, 0, 0, :] = torch.tensor([0.6, 0.5, 0.25, 0.09, 0.8, 
    #                                       0.8, 0.4, 0.4, 0.6, 0.9, 
    #                                       0.1, 0.0])
    
    # target_grid[0, 0, 0, :] = torch.tensor([0.5, 0.5, 0.16, 0.16, 1.0,
    #                                         0.0, 0.0, 0.0, 0.0, 0.0,
    #                                         1.0, 0.0])
    

    # pred_grid[0, 1, 1, :] = torch.tensor([0.6, 0.5, 0.25, 0.09, 0.8, 
    #                                       0.8, 0.4, 0.4, 0.6, 0.9, 
    #                                       0.1, 0.0])
    
    # target_grid[0, 1, 1, :] = torch.tensor([0.85, 0.4, 0.4, 0.55, 1.0,
    #                                         0.0, 0.0, 0.0, 0.0, 0.0,
    #                                         1.0, 0.0])

    # get_analysis_masks(pred_grid, target_grid)
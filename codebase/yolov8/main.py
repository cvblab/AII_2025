from codebase.yolov8.yolo_utils import get_detection_metrics
from codebase.data.dataset import get_model_paths
from yolo_utils import train_model

if __name__ == '__main__':

    train_model(name="yolov8_combined")  # train YOLO

    data = "aureus"  # aureus  dsb breast tcell flow_chamber
    mode = "test"
    yolo_path = '../logs/training/yolo/yolov8_combined/weights/best.pt'
    semantic_seg_model_path = ""
    input_type = "images" # binary_masks   predicted_masks  images

    #check model performance
    get_detection_metrics(data, mode, f"../{yolo_path}", semantic_seg_model_path, input_type=input_type)


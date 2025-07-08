from codebase.yolov8.yolo_utils import get_detection_metrics
from codebase.data.dataset import get_model_paths
from yolo_utils import train_model

if __name__ == '__main__':
    #train_model(name="yolov8_all")  # to yolov8x_dsb18 yolov8
    data = "aureus"  # aureus  dsb  mixed  breast subtilis neurips
    mode = "test"
    instance_seg_model_path, semantic_seg_model_path, yolo_path, cellpose_path = get_model_paths(data)
    yolo_path = '../logs/training/yolo/yolov8_all/weights/best.pt'
    input_type = "images" # binary_masks   predicted_masks  images
    get_detection_metrics(data, mode, f"../{yolo_path}", semantic_seg_model_path, input_type=input_type)

    # weights_path = "../../weights/yolo/yolov8n_dsb18.pt"  # yolov8n_dsb18 yolov8n_fluo yolov8x_dsb18 yolov8x_fluo
    # images_path = "../../datasets/fluorescence_dataset/test/fluorescence"
    # images_path = "../datasets/fluorescence_dataset/yolov8n_fluo/patches/fluorescence"
    # model = YOLO(weights_path)
    # predict_and_visualize(images_path, model)

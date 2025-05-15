from codebase.yolov8.yolo_utils import get_detection_metrics, predict_and_visualize, train_model


if __name__ == '__main__':
   #train_model(name="yolov8_a_masks")  # to yolov8x_dsb18 yolov8
    data = "dsb"  # aureus  dsb  mixed  breast subtilis
    mode = "test"
    yolo_weights_path = "../../weights/yolo/yolov8n_dsb18.pt"  # runs/yolov8_dsb_masks/weights/best.pt  # ../../weights/yolo/yolov8n_dsb18.pt
    semantic_seg_model_path = "../../logs/training/semantic/dsb/unet_final_epoch50.pth"
    input_type = "predicted_masks" # binary_masks   predicted_masks  images
    get_detection_metrics(data, mode, yolo_weights_path, semantic_seg_model_path, input_type=input_type)

    # weights_path = "../../weights/yolo/yolov8n_dsb18.pt"  # yolov8n_dsb18 yolov8n_fluo yolov8x_dsb18 yolov8x_fluo
    # images_path = "../../datasets/fluorescence_dataset/test/fluorescence"
    # images_path = "../datasets/fluorescence_dataset/yolov8n_fluo/patches/fluorescence"
    # model = YOLO(weights_path)
    # predict_and_visualize(images_path, model)

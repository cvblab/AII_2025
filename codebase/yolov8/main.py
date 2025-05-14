from codebase.yolov8.yolo_utils import get_detection_metrics, predict_and_visualize, train_model


if __name__ == '__main__':
   #train_model(name="yolov8_a_masks")  # to yolov8x_dsb18 yolov8
    data = "dsb"  # aureus  dsb  mixed  breast subtilis
    mode = "test"
    #weights_path = "../../weights/yolo/yolov8n_dsb18.pt"
    weights_path = "runs/yolov8_dsb_masks/weights/best.pt"
    input_type = "binary_masks" # binary_masks   predicted_masks  images
    get_detection_metrics(data, mode, weights_path,input_type="images")

    # weights_path = "../../weights/yolo/yolov8n_dsb18.pt"  # yolov8n_dsb18 yolov8n_fluo yolov8x_dsb18 yolov8x_fluo
    # images_path = "../../datasets/fluorescence_dataset/test/fluorescence"
    # images_path = "../datasets/fluorescence_dataset/yolov8n_fluo/patches/fluorescence"
    # model = YOLO(weights_path)
    # predict_and_visualize(images_path, model)

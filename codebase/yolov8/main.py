from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

def predict_and_visualize(folder_path, model):
    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(image_path)
        # Perform prediction on the image
        results = model.predict(source=image_path, save=False, save_txt=False)

        # Load the image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting

        # Get predictions
        boxes = results[0].boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)

        # Create a subplot with 1 row and 2 columns
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        # Plot the original image
        axs[0].imshow(image_rgb)
        axs[0].axis('off')
        axs[0].set_title("Original Image")

        # Plot the image with bounding boxes
        axs[1].imshow(image_rgb)
        ax = axs[1]
        # Add bounding boxes to the second subplot
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            # Draw rectangle (red color, no label or confidence)
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
        ax.axis('off')
        axs[1].set_title("Image with Bounding Boxes")
        plt.show()
        plt.close()


def train_model():
    model = YOLO("yolov8n.pt") #yolov8x.pt
    results = model.train(data="mydata.yaml", epochs=300)
    return results


if __name__ == '__main__':
    #train_model()  # to yolov8x_dsb18 yolov8
    weights_path = "../runs/detect/yolov8x_dsb18/weights/best.pt"  #yolov8n_dsb18 yolov8n_fluo yolov8x_dsb18 yolov8x_fluo
    images_path = "../datasets/fluorescence_dataset/test/fluorescence"
    #images_path = "../datasets/fluorescence_dataset/yolov8n_fluo/patches/fluorescence"
    model = YOLO(weights_path)
    predict_and_visualize(images_path, model)
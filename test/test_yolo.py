
from ultralytics import YOLO

# Test ONLY (not in requirements.txt)
import cv2

# img = '../tmp/robot-camera-FoJEMSFSMOTa38vL.jpg'
# img = '../tmp/robot-camera-qCGfvvZGZ0y6h5Cw.jpg'
# img = '../tmp/robot-camera-5RG0FB2DRhSMyT3l.jpg'
# img = '../tmp/1751172448649.jpg'
# img = '../tmp/1751172531893.jpg'
# img = '../tmp/1751175160019.jpg'
# img = '../tmp/robot-camera-l6oV9Kddyyxr39iY.jpg'
# img = '../train/images/000000000672.jpg'
img = '../tmp/robot-camera-1751264908448767095.jpg'


image = cv2.imread(img)
model = YOLO("../models/best.pt")
results = model.predict(img, show=True, save_txt=True)

for detection in results:
    boxes = detection.boxes.xyxy
    for (idx, box) in enumerate(boxes):
        x_min, y_min, x_max, y_max = box.tolist()

        # 绘制边界框
        cv2.rectangle(image,
                      (int(x_min), int(y_min)),
                      (int(x_max), int(y_max)),
                      (0, 255, 0), 2)  # 绿色框
        print(f"[{x_min}, {y_min}]-[{x_max}, {y_max}]")

        # 添加标签
        # label = f"{detection.names[idx]}: {detection.probs[idx]:.2f}"
        # cv2.putText(image, label,
        #             (int(x_min), int(y_min) - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 保存或显示结果
cv2.imwrite('detection_result.jpg', image)

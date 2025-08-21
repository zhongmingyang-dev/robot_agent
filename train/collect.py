import os

import grpc
import cv2
import numpy as np

import time

import proto.robot_control_pb2 as pb
import proto.robot_control_pb2_grpc as pbg

# 连接到 gRPC 服务器
channel = grpc.insecure_channel("192.168.12.210:50051")  # 替换为你的地址
stub = pbg.CameraServiceStub(channel)

# 获取相机配置
camera_config = stub.GetCameraConfig(pb.Empty())
print("Camera config:")
print(camera_config)

# 获取图像数据
response = stub.GetImage(pb.Empty())

# 解析图像数据（JPEG 格式）
image_bytes = response.full_image
np_arr = np.frombuffer(image_bytes, np.uint8)
image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# 保存图像
os.makedirs("images", exist_ok=True)
cv2.imwrite(f"images/saved_image-{time.time()}.jpg", image)
print("图像已保存为 saved_image.jpg")

# 可选：查看深度图信息
depth_info = response.depth_image
print(f"Min depth: {depth_info.min_depth} m, Max depth: {depth_info.max_depth} m")

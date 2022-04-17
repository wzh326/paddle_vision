import string
import cv2
import numpy as np
import time
import argparse

from Lib.PaddleSeg_Inference_Lib import Paddle_Seg
import rospy
from std_msgs.msg import Header
from cv_bridge import CvBridge , CvBridgeError
from Lib.PaddleSeg_Inference_Lib import Paddle_Seg
from sensor_msgs.msg import Image
# --------------------------配置区域--------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--infer_img_size',default=640, type=int,help='自定义模型预测的输入图像尺寸')
parser.add_argument("--use_gpu",default = False,type = bool,help='是否使用GPU')
parser.add_argument("--gpu_memory",default = 500,type = int,help='GPU的显存')
parser.add_argument("--use_tensorrt",default = False,type = bool,help='是否使用TensorRT')
parser.add_argument("--precision_mode",default = "fp32",type = string,help='TensorRT精度模式')
parser.add_argument("--model_folder_dir",default = "model/hardnet_test",type = string,help='模型文件路径')
parser.add_argument("--label_list",default =["sidewalk","other","blind_road"],type = list,help='类别信息')

args = parser.parse_args()
# -----------------------------------------------------------
        
if __name__ == '__main__':
    paddle_seg = Paddle_Seg(model_folder_dir=args.model_folder_dir, infer_img_size=args.infer_img_size, 
                            use_gpu=args.use_gpu, gpu_memory=args.gpu_memory, use_tensorrt=args.use_tensorrt, 
                            precision_mode=args.precision_mode,label_list=args.label_list)

    rospy.init_node('camera_node', anonymous=True) #定义节点
    image_pub=rospy.Publisher('/image_view/image_raw', Image, queue_size = 1) #定义话题
    while not rospy.is_shutdown():
        start = time.time()
        image = cv2.imread("0.jpg",1)    
        paddle_seg.init(image.shape[1],image.shape[0])        
        # 预测
        result = paddle_seg.infer(image)

        # 绘制结果
        frame, _ = paddle_seg.post_process(image, result)

        ros_frame = Image()
        header = Header(stamp = rospy.Time.now())
        header.frame_id = "Camera"
        ros_frame.header=header
        ros_frame.width = image.shape[0]
        ros_frame.height = image.shape[1]
        ros_frame.encoding = "bgr8"
        ros_frame.data = np.array(image).tobytes()#图片格式转换
        image_pub.publish(ros_frame) #发布消息

        end = time.time()  
        print("cost time:", end-start ) # 看一下每一帧的执行时间，从而确定合适的rate
        rate = rospy.Rate(25) # 10hz 
        cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
        cv2.imshow("image", frame)
        cv2.resizeWindow('image', 640, 480)
        cv2.waitKey(1)




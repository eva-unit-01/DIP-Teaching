import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    warped_image = np.full(image.shape,fill_value=255,dtype=np.uint8)
    source_pts = np.array(source_pts,dtype=np.float32)
    target_pts = np.array(target_pts,dtype=np.float32)
    #由坐标系标记方式转换为矩阵标记方式
    source_pts = np.column_stack((source_pts[:,1],source_pts[:,0]))
    target_pts = np.column_stack((target_pts[:,1],target_pts[:,0]))
    source_pts, target_pts = target_pts, source_pts  # 交换源点和目标点，使得目标点为控制点，源点为目标点

    ### FILL: 基于MLS or RBF 实现 image warping
    ### MLS
    width = image.shape[1]
    height = image.shape[0]
    numControlPoints = source_pts.shape[0]

    for j in range(width):
        for i in range(height):
            #NOTICE: 在image(i,j) 中是以矩阵方式表示，矩阵第i行第j列；
            #          在source_pts中，以坐标系方式表示，控制点坐标(i,j)代表的第j行第i列，两者表示方式有区别。一致以image方式为准。
            pointWeights = np.linalg.norm([i,j]-source_pts,axis=1)+eps
            pointWeights = pointWeights**(-2*alpha)
    
            totalWeight = np.sum(pointWeights)
            pStar = pointWeights @ source_pts / totalWeight
            pHat = source_pts - pStar
            pHatVertical = np.column_stack((-pHat[:,1],pHat[:,0]))

            qStar = pointWeights @ target_pts / totalWeight
            qHat = target_pts - qStar
            qHatVertical = np.column_stack((-qHat[:,1],qHat[:,0]))

            w1 = np.sum((pointWeights[:,np.newaxis] * pHat)*qHat)
            w2 = np.sum((pointWeights[:,np.newaxis] * qHat)*pHatVertical)
            
            muR = np.sqrt(w1**2 + w2**2)
            M = np.zeros((2,2))
            for k in range(numControlPoints):
                M += pointWeights[k] * np.vstack((pHat[k],-pHatVertical[k]))@np.hstack((qHat[k].reshape(-1,1),-qHatVertical[k].reshape(-1,1)))/muR

            dest = ([i,j]-pStar)@M + qStar

            dest[0] = max(0,min(dest[0],height-1))
            dest[1] = max(0,min(dest[1],width-1))

            lowX = int(np.ceil(dest[0]))-1
            lowY = int(np.ceil(dest[1]))-1
            highX = int(np.ceil(dest[0]))
            highY = int(np.ceil(dest[1]))

            lowX = max(0,lowX)
            lowY = max(0,lowY)
            highX = min(height-1,highX)
            highY = min(width-1,highY)

            a = dest[0] - lowX
            b = dest[1] - lowY

            warped_image[i,j] = (1-a)*(1-b)*image[lowX,lowY] + a*(1-b)*image[highX,lowY] + (1-a)*b*image[lowX,highY] + a*b*image[highX,highY]


    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()

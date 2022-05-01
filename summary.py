#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.yolo import yolo_body
from utils.utils import net_flops

if __name__ == "__main__":
    input_shape     = [640, 640, 3]
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes     = 80
    phi             = 'l'

    model = yolo_body(input_shape, anchors_mask, num_classes, phi)
    #--------------------------------------------#
    #   查看网络结构网络结构
    #--------------------------------------------#
    model.summary()
    #--------------------------------------------#
    #   计算网络的FLOPS
    #--------------------------------------------#
    net_flops(model, table=False)
    
    #--------------------------------------------#
    #   获得网络每个层的名称与序号
    #--------------------------------------------#
    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)

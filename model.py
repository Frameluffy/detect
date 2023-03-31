import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    # 加载经过预训练的模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    # 获取分类器的输入参数的数量in_features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    print("in_features:", in_features)
    # replace the pre-trained head with a new one
    # 用新的头部替换预先训练好的头部
    # 本实验的num_classes为3 
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


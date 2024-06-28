
import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO

if __name__ == "__main__":
    '''
    Recall and Precision are not concepts of area like AP, so the Recall and Precision values of the network are different at different thresholds (Confidence).
    By default, the Recall and Precision calculated by this code represent the Recall and Precision values corresponding to a threshold (Confidence) of 0.5.

    Due to the limitations of the mAP calculation principle, the network needs to obtain almost all prediction boxes when calculating the mAP, so that the Recall and Precision values under different threshold conditions can be calculated.
    Therefore, the number of boxes in the txt files in map_out/detection-results/ obtained by this code is generally more than that obtained by direct prediction. The purpose is to list all possible prediction boxes.
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode is used to specify the content calculated when this file runs.
    #   map_mode=0 represents the entire mAP calculation process, including obtaining prediction results, obtaining ground truth boxes, and calculating VOC_map.
    #   map_mode=1 represents only obtaining prediction results.
    #   map_mode=2 represents only obtaining ground truth boxes.
    #   map_mode=3 represents only calculating VOC_map.
    #   map_mode=4 represents using the COCO toolbox to calculate the 0.50:0.95map of the current dataset. Prediction results and ground truth boxes need to be obtained first, and pycocotools needs to be installed.
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #--------------------------------------------------------------------------------------#
    #   classes_path is used to specify the classes for which VOC_map needs to be measured.
    #   Generally, it should be consistent with the classes_path used for training and prediction.
    #--------------------------------------------------------------------------------------#
    classes_path    = 'model_data/label-x.txt'
    #--------------------------------------------------------------------------------------#
    #   MINOVERLAP is used to specify the mAP0.x to be obtained. For the meaning of mAP0.x, please search online.
    #   For example, to calculate mAP0.75, set MINOVERLAP = 0.75.
    #
    #   When the overlap between a predicted box and a ground truth box is greater than MINOVERLAP, the predicted box is considered a positive sample; otherwise, it is a negative sample.
    #   Therefore, the larger the value of MINOVERLAP, the more accurate the predicted box needs to be to be considered a positive sample, and the lower the calculated mAP value.
    #--------------------------------------------------------------------------------------#
    MINOVERLAP      = 0.5
    #--------------------------------------------------------------------------------------#
    #   Due to the limitations of the mAP calculation principle, the network needs to obtain almost all prediction boxes when calculating the mAP.
    #   Therefore, the confidence value should be set as low as possible to obtain all possible prediction boxes.
    #   
    #   This value is generally not adjusted. Because calculating the mAP requires obtaining almost all prediction boxes, the confidence value above should not be changed casually.
    #   To obtain Recall and Precision values under different thresholds, please modify the score_threshold below.
    #--------------------------------------------------------------------------------------#
    confidence      = 0.001
    #--------------------------------------------------------------------------------------#
    #   The size of the non-maximum suppression value used during prediction. The larger the value, the less strict the non-maximum suppression.
    #   
    #   This value is generally not adjusted.
    #--------------------------------------------------------------------------------------#
    nms_iou         = 0.5
    #---------------------------------------------------------------------------------------------------------------#
    #   Recall and Precision are not concepts of area like AP, so the Recall and Precision values of the network are different at different thresholds.
    #   
    #   By default, the Recall and Precision calculated by this code represent the Recall and Precision values corresponding to a threshold of 0.5 (defined here as score_threshold).
    #   Because calculating the mAP requires obtaining almost all prediction boxes, the confidence value above should not be changed casually.
    #   Here, a score_threshold is specifically defined to represent the threshold, so that the Recall and Precision values corresponding to the threshold can be found when calculating the mAP.
    #---------------------------------------------------------------------------------------------------------------#
    score_threshold = 0.5
    #-------------------------------------------------------#
    #   map_vis is used to specify whether to enable visualization of VOC_map calculation.
    #-------------------------------------------------------#
    map_vis         = False
    #-------------------------------------------------------#
    #   Path to the VOC dataset.
    #   Default path points to the VOC dataset in the root directory.
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #-------------------------------------------------------#
    #   Output folder for results, default is map_out.
    #-------------------------------------------------------#
    map_out_path    = 'map_out'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence = confidence, nms_iou = nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threshold = score_threshold, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
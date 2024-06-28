
import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO

if __name__ == "__main__":
    '''
    Recall and Precision are not like AP, which is an area concept, so the network's Recall and Precision values are different when the threshold (Confidence) changes.
    By default, the Recall and Precision calculated by this code represent the Recall and Precision values when the threshold (Confidence) is 0.5.

    Due to the limitation of the mAP calculation principle, the network needs to obtain almost all prediction boxes when calculating mAP so that it can calculate the Recall and Precision values under different threshold conditions.
    Therefore, the number of boxes in the txt files in map_out/detection-results/ obtained by this code is generally more than the direct prediction, aiming to list all possible prediction boxes.
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode is used to specify the content calculated when this file is run
    #   map_mode 0 represents the entire mAP calculation process, including obtaining prediction results, obtaining ground truth, and calculating VOC_map.
    #   map_mode 1 represents only obtaining prediction results.
    #   map_mode 2 represents only obtaining ground truth.
    #   map_mode 3 represents only calculating VOC_map.
    #   map_mode 4 represents using the COCO toolbox to calculate the 0.50:0.95 map of the current dataset. This requires obtaining prediction results, obtaining ground truth, and installing pycocotools.
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #--------------------------------------------------------------------------------------#
    #   The classes_path here is used to specify the categories for which VOC_map needs to be measured.
    #   Generally, it is consistent with the classes_path used for training and prediction.
    #--------------------------------------------------------------------------------------#
    classes_path    = 'model_data/label.txt'
    #--------------------------------------------------------------------------------------#
    #   MINOVERLAP is used to specify the desired mAP0.x. Please search for the meaning of mAP0.x.
    #   For example, to calculate mAP0.75, you can set MINOVERLAP = 0.75.
    #
    #   When a predicted box overlaps with the ground truth box by more than MINOVERLAP, the predicted box is considered a positive sample; otherwise, it is a negative sample.
    #   Therefore, the higher the MINOVERLAP value, the more accurate the predicted box needs to be to be considered a positive sample, resulting in a lower calculated mAP value.
    #--------------------------------------------------------------------------------------#
    MINOVERLAP      = 0.5
    #--------------------------------------------------------------------------------------#
    #   Due to the limitation of the mAP calculation principle, the network needs to obtain almost all prediction boxes when calculating mAP.
    #   Therefore, the confidence value should be set as low as possible to obtain all possible prediction boxes.
    #
    #   This value is generally not adjusted. Because calculating mAP requires obtaining almost all prediction boxes, the confidence defined here should not be changed arbitrarily.
    #   To obtain Recall and Precision values under different threshold values, please modify the score_threhold below.
    #--------------------------------------------------------------------------------------#
    confidence      = 0.001
    #--------------------------------------------------------------------------------------#
    #   The size of the non-maximum suppression value used during prediction, the larger it is, the less strict the non-maximum suppression.
    #
    #   This value is generally not adjusted.
    #--------------------------------------------------------------------------------------#
    nms_iou         = 0.5
    #---------------------------------------------------------------------------------------------------------------#
    #   Recall and Precision are not like AP, which is an area concept, so the network's Recall and Precision values are different when the threshold changes.
    #
    #   By default, the Recall and Precision calculated by this code represent the Recall and Precision values when the threshold is 0.5 (defined here as score_threhold).
    #   Because calculating mAP requires obtaining almost all prediction boxes, the confidence defined above should not be changed arbitrarily.
    #   A separate score_threhold is defined here to represent the threshold value, in order to find the Recall and Precision values corresponding to the threshold when calculating mAP.
    #---------------------------------------------------------------------------------------------------------------#
    score_threhold  = 0.5
    #-------------------------------------------------------#
    #   map_vis is used to specify whether to enable visualization of VOC_map calculation
    #-------------------------------------------------------#
    map_vis         = False
    #-------------------------------------------------------#
    #   Point to the folder where the VOC dataset is located
    #   The default points to the VOC dataset in the root directory
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #-------------------------------------------------------#
    #   Folder for result output, default is map_out
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
        get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO

if __name__ == "__main__":
    '''
    Recall and Precision are not area concepts like AP, so the values of Recall and Precision are different when the threshold (Confidence) is different.
    By default, the Recall and Precision calculated by this code represent the Recall and Precision values when the threshold (Confidence) is 0.5.
    
    Due to the limitations of the mAP calculation principle, the network needs to obtain almost all the prediction boxes when calculating the mAP, so that the Recall and Precision values under different threshold conditions can be calculated.
    Therefore, the number of boxes in the txt files in map_out/detection-results/ obtained by this code is generally more than directly predicted, in order to list all possible prediction boxes.
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode is used to specify the content to be calculated when this file is run.
    #   map_mode 0 means the entire map calculation process, including obtaining prediction results, obtaining ground truth boxes, and calculating VOC_map.
    #   map_mode 1 means only obtaining prediction results.
    #   map_mode 2 means only obtaining ground truth boxes.
    #   map_mode 3 means only calculating VOC_map.
    #   map_mode 4 means calculating the current dataset's 0.50:0.95 map using the COCO toolbox. Prediction results and ground truth boxes need to be obtained and pycocotools needs to be installed.
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode = 0
    #--------------------------------------------------------------------------------------#
    #   The classes_path here is used to specify the classes for which VOC_map is to be measured.
    #   Generally, it is the same as the classes_path used for training and prediction.
    #--------------------------------------------------------------------------------------#
    classes_path = 'model_data/voc_classes.txt'
    #--------------------------------------------------------------------------------------#
    #   MINOVERLAP is used to specify the desired mAP0.x. Please search online to understand the significance of mAP0.x.
    #   For example, to calculate mAP0.75, set MINOVERLAP = 0.75.
    #
    #   When a prediction box overlaps with the ground truth box by more than MINOVERLAP, the prediction box is considered a positive sample; otherwise, it is a negative sample.
    #   Therefore, the larger the value of MINOVERLAP, the more accurate the prediction box must be to be considered a positive sample, and the lower the calculated mAP value will be.
    #--------------------------------------------------------------------------------------#
    MINOVERLAP = 0.5
    #--------------------------------------------------------------------------------------#
    #   Visualization flag, used to visualize the predicted and ground truth boxes in images
    #--------------------------------------------------------------------------------------#
    map_vis = False
    #-------------------------------------------------------#
    #   Path to the dataset
    #-------------------------------------------------------#
    VOCdevkit_path = 'VOCdevkit'

    #--------------------------------------------------------------------------------------#
    #   The score_threhold is used to filter out low-confidence prediction boxes when calculating mAP.
    #--------------------------------------------------------------------------------------#
    score_threhold = 0.5
    #-------------------------------------------------------#
    #   Get classes and number of classes
    #-------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    
    #-------------------------------------------------------#
    #   Load YOLO model, defaults to using gpu
    #-------------------------------------------------------#
    yolo = YOLO()

    #-------------------------------------------------------#
    #   Generate folder to store results
    #-------------------------------------------------------#
    map_out_path = 'map_out'
    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()
    
    if map_mode == 0 or map_mode == 1:
        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image = Image.open(image_path)
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
                    if obj.find('difficult') != None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold=score_threhold, path=map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names=class_names, path=map_out_path)
        print("Get map done.")

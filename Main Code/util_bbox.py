from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
import torch


def box_area(box):
    return (box[3]-box[1]) * (box[2] - box[0])


def inside(b,b2):

    '''
    checks if box b2 is inside box b
    :param b,b2 = boxes
    :return: bool
    '''

    return b[0] <= b2[0] and b[1] <= b2[1] and b[2] >= b2[2] and b[3] >= b2[3]


def isClose(b,b2):

    # box has the format [row_min, col_min, row_max, col_max]
    A = 7000
    A_i = 400
    len_thresh = 70
    near_thresh = 30
    side_thresh = 40
    if box_area(b) > A and box_area(b2) > A:
        return False
    '''if inside(b,b2) or inside(b2,b):
        return False'''
    '''if abs(myarea(b) - myarea(b2)) > A_i:
     return False'''
    if abs((b[3]-b[1]) - (b2[3]-b2[1])) > side_thresh and abs((b[2]-b[0]) - (b2[2]-b2[0])) > side_thresh:
        return False
    if (abs(b[3]-b2[1]) < near_thresh and abs(b[0] - b2[0]) < len_thresh and abs(b[2] - b2[2]) < len_thresh) or (abs(b[2] - b2[0]) < near_thresh and abs(b[1] - b2[1]) < len_thresh and abs(b[3] - b2[3]) < len_thresh):
        return True
    if (abs(b2[3]-b[1]) < near_thresh and abs(b2[0] - b[0]) < len_thresh and abs(b2[2] - b[2]) < len_thresh) or (abs(b2[2] - b[0]) < near_thresh and abs(b2[1] - b[1]) < len_thresh and abs(b2[3] - b[3]) < len_thresh):
        return True

    return False


def sketch2box(filename, w=0, h=0):

    im = plt.imread(filename)
    wr = w/im.shape[1] if w != 0 else 1
    hr = h/im.shape[0] if h != 0 else 1
    minArea = (im.shape[0] // 90) ** 2
    blobs = 255 * (im > 150)
    labels, num  = measure.label(blobs, connectivity=2, return_num =True)
    #num = labels.max()
    boxes = [[]]

    for i in range(1, num):
        zipped = np.where(labels == i)

        b = [i, min(zipped[0]), min(zipped[1]), max(zipped[0]), max(zipped[1])]
        # 0 = height axis
        # 1 = width axis
        # b = [label, row_min,col_min,row_max,col_max]
        area = (b[4] - b[2] + 1) * (b[3] - b[1] + 1)
        if area < minArea: continue
        boxes.append(b)

    del boxes[0]
    boxs2 = [i[1:] for i in boxes]  # won't have the label

    again = True

    while (again):

        combox = []
        again = False
        for i, b in enumerate(boxs2):
            # print(i,b,myarea(b))
            for b2 in boxs2[i + 1:]:
                if isClose(b, b2):
                    again = True
                    combox = [min(b[0], b2[0]), min(b[1], b2[1]), max(b[2], b2[2]), max(b[3], b2[3])]
                    boxs2.remove(b2)
                    break
            if again:
                boxs2.remove(b)
                boxs2.insert(0, combox)
                break

    # box had the format [row_min, col_min, row_max, col_max]
    boxs2 = [[j_box[1], j_box[0], j_box[3], j_box[2]] for j_box in boxs2]
    # box has the format [col_min, row_min, col_max, row_max]

    boxs2.append([0, 0, im.shape[1], im.shape[0]])

    return norm_it(boxs2, wr, hr)


def image_boxes(img, boxes):

    img2 = img.copy()
    for i in boxes:
        img2 = image_box(img2,i)

    return img2


def image_box(img, box, label=' ', colour='blue'):
    '''
    :param img:     an image on whom I'm finding the box
    :param box:     [x_leftTop,y_leftTop,x_rightBottom,y_rightBottom]   (col1,row1,col2,row2)
    :param label :  label of the corresponding box
    :return:        image with the boxes on them  (gt = green, prediction box = blue)
    '''

    img2 = img.copy()  # returning the image separately
    draw = ImageDraw.Draw(img2)
    draw.rectangle(box, fill=None, outline=colour, width=2)
    font_size = min((box[3] - box[1]) // 2, 20)
    font = ImageFont.truetype("arial.ttf", font_size)
    draw.text((box[0] + 5, box[1] + 5), text=label, fill=colour, font=font)  # labelling GT

    return img2


def boxFinder(img, gt_box, pred_box, label):
    '''
    :param img:     an image on whom I'm finding the box
    :param box:     [x_leftTop,y_leftTop,x_rightBottom,y_rightBottom]   (col1,row1,col2,row2)
    :param gt_box : same box with ground-truth information
    :param label :  label of the corresponding box
    :return:        image with the boxes on them  (gt = green, prediction box = blue)
    '''

    img2 = image_box(img, label=label, box=gt_box, colour='green')
    img2 = image_box(img2, label=label, box=pred_box, colour='red')
    '''draw = ImageDraw.Draw(img2)
    draw.rectangle(gt_box, fill=None, outline='green', width=2)
    draw.rectangle(box, fill=None, outline='blue', width=2)
    fontSize = min((gt_box[3] - gt_box[1]) // 2, 20)
    font = ImageFont.truetype("arial.ttf", fontSize)
    draw.text((gt_box[0] + 5, gt_box[1] + 5), text=label, fill='green', font=font)  # labelling GT
    draw.text((box[0] + 5, box[3] - (fontSize + 5)), text=label, fill='blue', font=font)  # labelling prediction
    '''
    return img2


def treeBuilder(nodes):
    '''
    :param nodes: A tensor of Nx4 that has box coordinates of N nodes in a picture.
                  (col1,row1,col2,row2)
    :return: tensor[col1,row1,col2,row2, area, self_id, parent_id]
    '''
    b = nodes.clone()

    b = torch.cat([b, torch.zeros(b.shape[0], 3).type(torch.int64)], dim=1)
    # torch.zeros(b.size(dim=0), 3).type(torch.LongTensor)
    # increasing 3 more col. to store area, index and parentID per box

    for i, a in enumerate(b, 0):
        a[4] = (a[2] - a[0]) * (a[3] - a[1])  # calculating area
        a[5] = i  # storing original node index

    # Sorting the tensor in the format of decreasing order of area
    b = b[b[:, 4].argsort(descending=True), :]
    '''
    ? Doesn't mean that a smaller box will always be under a larger box.

    ? so I need to calculate the IOU score.
    ? that way I can predict the children structure

    = arrange bounding boxes in decreasing order of their area 
    = for each bbox Bi, check all other boxes of lower area whose co-ordinates fall under them. X
    = for each box Bi, check if any box with lower area, falls under it or not.
    =   = if its so, then take that box as the Bi and repeat the action recursively and exhaustively.
    = 
    = classify each such box as a child.
    '''

    # Creating the hierarchy :
    for c in range(1, b.shape[0]):
        par = b[c - 1]
        for k in b[c:]:
            if inside(par, k):   #k[0] >= par[0] and k[1] >= par[1] and k[2] <= par[2] and k[3] <= par[3]:
                k[6] = par[5]  # 6-> child index; 5-> parent index

    # print(b)
    return b


def norm_it(boxes, wr, hr):
    # [w0 h0 w1 h1]
    # wr = width ratio  = new_image_width/original_image_width
    # hr = width ratio  = new_image_height/original_image_height
    return [[int(i_box[0] * wr), int(i_box[1] * hr), int(i_box[2] * wr), int(i_box[3] * hr)] for i_box in boxes]


def give_json_boxes(filename, w=0, h=0, return_Label=False):
    # Accepts a json filename
    # normalised height,weight dimensions
    # Getting all NxD boxes
    # w = new image width = 0 means no normalising
    # h = new image height
    h_ratio = h / 2560 if h != 0 else 1
    w_ratio = w / 1440 if w != 0 else 1
    with open(filename, 'r') as file:
        jf = json.load(file)

    arr_Nxd = [[0, 0, 1440, 2560]]
    label = ['Base']
    for child in jf['children']:
        bounder(child, arr_Nxd, label)

    if return_Label:
        return norm_it(arr_Nxd, w_ratio, h_ratio), label
    else:
        return norm_it(arr_Nxd, w_ratio, h_ratio)


def bounder(file, arr_Nxd, label):
    '''
    This function gets you the bounding box and its label from a json and gives an index in depth-first order search
    :param file: a json file or basically a dictionary
    :return: an array of Nx5. First 4 is the bounding box and the last is the component label
    '''
    arr_Nxd.append(file['bounds'])
    label.append(file['componentLabel'])

    if 'children' in file.keys():
        for k in file['children']:
            bounder(k, arr_Nxd, label)

    # return arr_Nxd


def giveImg(filename):
    # creates an image describing the graphical heirarchy of nodes and components

    # Getting all NxD boxes
    # label = ['Base']
    arr_Nxd, label = give_json_boxes(filename, 0, 0, return_Label=True)  # not normalising.

    '''
    for i in jf['children']:
        bounder(i, arr_Nxd, label)

    for i, d in enumerate(arr_Nxd):
        print(d, '\t', label[i])
    print(len(arr_Nxd), len(label))
    '''
    # order of arr_Nxd is the order of labels and are ordered/indexed in Root-Left-Right traversal order.

    tnsor_Nxd = torch.tensor(arr_Nxd)
    img = Image.new('RGB', (1440, 2560))
    draw = ImageDraw.Draw(img)

    ans = treeBuilder(tnsor_Nxd)

    # print('the graph is below:\n')
    for i in ans:
        # print(i, '\t', label[i[5].item()])

        pnts = [int(m.item()) for m in i[:4]]

        border = 5
        # positioning the text
        fontSize = min((pnts[3] - pnts[1]) // 2, 20)
        font = ImageFont.truetype("arial.ttf", fontSize)  # int((i[2].item()-i[0].item())/4)

        '''while img.crop([pnts[0]+5, pnts[1]+border, pnts[0]+20, pnts[1]+border+20]).getcolors()[0][1] != (0, 0, 0):
            border += 25'''
        draw.text((pnts[0] + 5, pnts[1] + border),
                  text=label[i[5].item()] + ':' + str(i[5].item()) + '-' + str(i[6].item()),
                  fill='white',
                  font=font)

        draw.rectangle(pnts, fill=None, outline='white', width=2)

    # img = img.resize((1080,1920))
    # img.show()
    return img


if __name__ == '__main__':

    filename = '/vol/research/sketchCV/SWIRE_Project/SWIRE_Data/Sketch/1_61641_2.jpg'
    im24 = plt.imread(filename)
    mybox = sketch2box(filename,0,0)
    ans = Image.fromarray(np.uint8(im24))
    ans = image_boxes(ans, mybox)
    ans.show()

    mybox = sketch2box(filename, 256,256)
    ans = Image.fromarray(np.uint8(im24))
    ans = ans.resize((256,256))
    ans = image_boxes(ans, mybox)
    ans.show()
import numpy as np
from bresenham import bresenham
import scipy.ndimage
# from util_bbox import image_boxes
# from PIL import Image


def mydrawPNG(vector_images, Side = 256,maxpixlen=256*256.0,per=1.0):
    raster_images, Sample_len, stroke_bbox = [], [], []
    for vector_image in vector_images:
        pixel_length = 0

        Sample_len = []
        raster_images = []
        raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)
        initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])

        stroke_bbox = []
        stroke_cord_buffer = []
        for i in range(0, len(vector_image)):
            if i > 0:
                if vector_image[i-1, 2] == 1:
                    initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

            cordList = list(bresenham(initX, initY, int(vector_image[i,0]), int(vector_image[i,1])))
            pixel_length += len(cordList)
            stroke_cord_buffer.extend([list(i) for i in cordList])

            for cord in cordList:
                if (cord[0] > 0 and cord[1] > 0) and (cord[0] < Side and cord[1] < Side):
                    raster_image[cord[1], cord[0]] = 255.0
            initX , initY = int(vector_image[i, 0]), int(vector_image[i,1])

            if pixel_length > per*maxpixlen  and vector_image[i, 2] == 1:
                pixel_length = len(np.where(vector_image[i+1:]==1)[0])
                # calculating number of strokes left after the 'per'% is reached.
                break

            if  vector_image[i, 2] == 1:
                min_x = np.array(stroke_cord_buffer)[:, 0].min()
                min_y = np.array(stroke_cord_buffer)[:, 1].min()
                max_x = np.array(stroke_cord_buffer)[:, 0].max()
                max_y = np.array(stroke_cord_buffer)[:, 1].max()
                stroke_bbox.append([min_x, min_y, max_x, max_y])
                stroke_cord_buffer = []

        raster_images.append(scipy.ndimage.binary_dilation(raster_image) * 255.0)
        Sample_len.append(pixel_length)
        #image_boxes(Image.fromarray(raster_images[-1]).convert('RGB'), stroke_bbox).show()
    return raster_images, Sample_len, stroke_bbox


def Preprocess_QuickDraw_redraw(vector_images, side = 256.0):
    vector_images = vector_images.astype(np.float)
    vector_images[:, :2] = vector_images[:, :2] / np.array([256, 256])
    vector_images[:,:2] = vector_images[:,:2] * side
    vector_images = np.round(vector_images)
    return vector_images

def redraw_Quick2RGB(vector_images, maxpixlen=256*256.0, per=1.0):
    vector_images_C = Preprocess_QuickDraw_redraw(vector_images)
    raster_images, Sample_len, strok_bbox = mydrawPNG([vector_images_C],maxpixlen=maxpixlen,per=per)
    return raster_images,  Sample_len, strok_bbox
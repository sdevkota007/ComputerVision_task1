from PIL import Image
import numpy as np
import math


def openImage(image, loc = None):
    '''
    reads an image with the given and returns a numpy array
    :param image:
    :return:
    '''
    img = Image.open(image)
    arr = np.asarray(img, dtype=np.int32)
    return arr

def saveImage(image_array, name, loc = None):
    '''
    saves the image array 'image_array' as image at the given location 'loc' with the given name
    :param image_array:
    :param name:
    :return:
    '''
    im = Image.fromarray(image_array.astype('uint8'), mode="RGB")
    im.save(name)

def showImage(image_array):
    '''

    :param image_array: a numpy array
    :return:
    '''
    im = Image.fromarray(image_array.astype('uint8'), mode="RGB")
    im.show()


def affineTransform(img_arr, M, size_new_img):
    '''
    used for scaling and transformation
    :param img_arr: numpy array of input image
    :param M: transformation matrix
    :param size_new_img: a tuple which contains the width and height of new image
    :return: resulting array when input image array is transformed with the supplied transformation matrix M
    '''

    # get height and width of old and new image
    h_old, w_old, num_channel = img_arr.shape
    w_new, h_new = size_new_img
    new_img = np.zeros((h_new, w_new, num_channel), dtype=np.uint8)
    for i in range(w_new):
        for j in range(h_new):
            # calculate pixel pos of destination pixel from pixel pos of source pixels
            pos_in_old_img = M.reshape((2,3))*(np.matrix([i,j,1]).reshape(3,1))
            pos_in_old_img = pos_in_old_img.reshape((1,2)).astype(int)
            pos_x, pos_y = pos_in_old_img.item(0), (pos_in_old_img.item(1))

            if 0<pos_x<w_old and 0<pos_y<h_old:
                new_img[j,i] = img_arr[pos_y, pos_x]

    return new_img


def affineTransformForRotation(img_arr, M, size_new_img):
    '''
    used for rotation
    :param img_arr: numpy array of input image
    :param M: transformation matrix
    :param size_new_img: a tuple which contains the width and height of new image
    :return: resulting array when input image array is transformed with the supplied transformation matrix M
    '''
    h_old, w_old, num_channel = img_arr.shape
    w_new, h_new = size_new_img
    new_img = np.zeros((h_new, w_new, num_channel), dtype=np.uint8)
    for i in range(w_new):
        for j in range(h_new):
            # calculate pixel pos of destination pixel from pixel pos of source pixels
            pos_in_old_img = ( M *
                               (
                                       np.matrix([i,j]).reshape((2,1)) \
                                       - np.matrix([w_new/2, h_new/2]).reshape((2,1)) \
                              ) \
                             )\
                             + np.matrix([w_old/2, h_old/2]).reshape((2,1))
            pos_in_old_img = pos_in_old_img.reshape((1,2)).astype(int)
            pos_x, pos_y = pos_in_old_img.item(0), (pos_in_old_img.item(1))

            if 0<pos_x<w_old and 0<pos_y<h_old:
                new_img[j,i] = img_arr[pos_y, pos_x]

    return new_img


def rotation(img_arr, theta):
    '''
    rotates the image by the given angle 'theta'
    :param img_arr: numpy array of imput image
    :param theta: angle of rotation
    :return: numpy array of rotated image
    '''
    # determine shape of img
    height, width = img_arr.shape[:2]

    cos = math.cos(math.radians(theta))
    sin = math.sin(math.radians(theta))

    # make rotation matrix
    rot_mat = np.matrix([[cos, sin], [-sin, cos]])

    # calculate new height and width to account for rotation
    newWidth = int((height * abs(sin)) + (width * abs(cos)))
    newHeight = int((height * abs(cos)) + (width * abs(sin)))

    new_img = affineTransformForRotation(img_arr, rot_mat, (newWidth, newHeight))
    return new_img



def scaling(img_arr, scale):
    '''
    scales the images by the given scale factor 'scale'
    :param img_arr: numpy array of imput image
    :param scale: scale-factor
    :return: numpy array of rotated image
    '''
    # shape of img
    height, width = img_arr.shape[:2]

    # make scaling matrix
    transformMat = np.matrix([[(1/scale), 0.0, 0.0],[0.0, (1/scale), 0.0]])

    # calculate new height and width
    newWidth = int(width*scale)
    newHeight = int(height*scale)

    new_img = affineTransform(img_arr, transformMat, (newWidth, newHeight))
    return new_img


def shift(img_arr, tx, ty):
    '''
    translated the image by the given shifting distance tx,ty
    :param img_arr: numpy array of imput image
    :param tx: shifting distance in x-direction
    :param ty: shifting distance in y-direction
    :return: numpy array of rotated image
    '''
    # shape of img
    height, width = img_arr.shape[:2]
    #print(height, width)
    transformMat = np.matrix([[1, 0, -tx],[0, 1, -ty]])

    # calculate new height and width
    #newWidth = width +abs(tx)
    #newHeight = height +abs(ty)
    newWidth = width
    newHeight = height

    new_img = affineTransform(img_arr, transformMat, (newWidth, newHeight))
    return new_img


def main():
    img_arr = openImage("image.png")
    #**************************************************

    #**************************************************
    #For rotation of Image
    theta = -45
    print("Rotating image by theta: {}".format(theta))
    rot_img = rotation(img_arr, theta)
    saveImage(rot_img, "output1.png")
    print("Done!!!")
    showImage(rot_img)
    #**************************************************


    #**************************************************
    # For Scaling of Image
    scale = 2
    print("Scaling image by factor: {}".format(scale))
    scaled_img = scaling(img_arr, scale)
    saveImage(scaled_img, "output2.png")
    print("Done!!!")
    showImage(scaled_img)
    #**************************************************

    #**************************************************
    #For Shifting of Image
    tx=10
    ty=10
    print("Shifting Image by tx={0}, ty={1}".format(tx, ty))
    shifted_img = shift(img_arr, tx, ty)
    saveImage(shifted_img, "output3.png")
    print("Done!!!")
    showImage(shifted_img)
    #**************************************************



if __name__ == '__main__':
    main()
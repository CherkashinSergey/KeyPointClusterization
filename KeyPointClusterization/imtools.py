#file imtools.py to
#store some of these generally useful routines

from PIL import Image
import os
from numpy import *
from pylab import *

def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) 
            if f.endswith('.jpg')]

def imresize(im, sz):
    """ Resize an image array using PIL. """
    pil_im = Image.fromarray(uint8(im))
    
    return array(pil_im.resize(sz))


# выравнивание гистограмм
def histeq(im, nbr_bins = 256):

    imhist, bins = histogram(im.flatten(), nbr_bins, normed = True)       

    cdf = imhist.cumsum()
    cdf = 255 * cdf/cdf[-1]

    im2 = interp(im.flatten(), bins[:-1], cdf)

    return im2.reshape(im.shape), cdf 

def normalizeHistogram(hist):
    divisor = max(hist)
    temp = [0.0 for i in range(len(hist))]
    if divisor == 0:
        return hist
    for i in range(len(hist)):
        temp[i] = float(hist[i]) / divisor
    return temp

def normArray(M):
    row_sums = M.sum(axis=1)
    return M / row_sums

'''
def normilizeHistogram2(imhist):      

    cdf = imhist.cumsum()
    cdf = 255 * cdf/cdf[-1]
    l = len(imhist)+1
    x = []
    for i in range(l):
        x.append(i)

    normHist = interp(imhist, x, cdf)
    
    return normHist
 '''

# усреднение изображений
def compute_average(imlist):
    averageim = array(Image.open(imlist[0]), 'f')
    skipped = 0
    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname))
            skipped += 1
        except:
            print imname + '... miss'
    averageim /= len(imlist)
            
    averageim /= (len(imlist) - skipped)

    return array(averageim, 'uint8')


#

def plot_2D_boundary(plot_range, points, decisionfcn, labels, values=[0]):
    clist = ['b','r','g','k','m','y']

    x = arange(plot_range[0], plot_range[1],.1)
    y = arange(plot_range[2], plot_range[3],.1)
    xx, yy = meshgrid(x,y)
    xxx, yyy = xx.flatten(), yy.flatten()
    zz = array(decisionfcn(xxx, yyy))
    zz = zz.reshape(xx.shape)

    contour(xx,yy,zz,values)

    for i in range(len(points)):
        d = decisionfcn(points[i][:,0], points[i][:,1])
        correct_ndx = labels[i] == d
        incorrect_ndx = labels[i] != d
        plot(points[i][correct_ndx,0],points[i][correct_ndx,1],'*',color=clist[i])
        plot(points[i][incorrect_ndx,0],points[i][incorrect_ndx,1],'o',color=clist[i])
   
    axis('equal')

def Htransform(im, H, out_size):
    """Applies a homography transform to im"""
    pil_im = Image.fromarray(im)
    pil_size = out_size[1], out_size[0]
    return array(pil_im.transform(pil_size, Image.PERSPECTIVE, H.reshape(9)[0:8] / H[2,2], Image.LINEAR))

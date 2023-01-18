import os
import sys
import math
import cv2
import numpy as np
import sympy as sp
from IPython.display import display
from matplotlib import pyplot as plt

import P0

def gaussian(x_, sigma=1, order=0):
  x = sp.Symbol('x')
  gaussian = sp.exp(-(x**2 / (2*sigma**2)))
  
  return sp.diff(gaussian, x, order).subs(x, x_).evalf()

def gaussianMask1D(sigma=0, sizeMask=0, order=0):
  # If sizeMask is 0, calculate it from sigma
  if sizeMask == 0:
    sizeMask = 2 * math.ceil(3 * sigma) + 1

  # If sigma is 0, calculate it from sizeMask
  elif sigma == 0:
    sigma = (sizeMask - 1) / 6

  # If sigma and sizeMask are 0, return error
  elif sigma == 0 and sizeMask == 0:
    print("Error: Either sigma or the size of the mask must be bigger than 0.")

  # Calculate k, the number of elements on each size
  k = int((sizeMask - 1) / 2)
  mask = np.empty(0)

  # Calculate the value of each point
  # Value obtained from the gaussian function multiplied by sigma to the power of the order
  for i in range(-k, k+1):
    mask = np.append(mask, gaussian(i, sigma, order) * sigma**order)

  # The sum of the mask must be 1
  if order == 0:
    mask = mask / np.sum(mask)

  return mask

def plotGraph(graph, title='No title'):
  plt.rcParams["figure.figsize"] = (8,6)
  x = np.arange(-(len(graph) - 1)/2, (len(graph) - 1)/2 + 1, 1.0)
  plt.plot(x, graph, label='σ = '+str(round(sigma, 2)))
  plt.title(title)
  plt.legend()
  plt.show()

def plotGaussian(sigma=0, sizeMask=0, order=0, title=''):
  # If sizeMask is 0, calculate it from sigma
  if sizeMask == 0:
    sizeMask = 2 * math.ceil(3 * sigma) + 1

  # If sigma is 0, calculate it from sizeMask
  elif sigma == 0:
    sigma = (sizeMask - 1) / 6

  # If sigma and sizeMask are 0, return error
  elif sigma == 0 and sizeMask == 0:
    print("Error: Either sigma or the size of the mask must be bigger than 0.")

  # Calculate k, the number of elements on each size
  k = int((sizeMask - 1) / 2)
  discrete = np.empty(0)

  # Calculate the value of each point
  # Value obtained from the gaussian function multiplied by sigma to the power of the order
  x_discrete = np.arange(-k, k+1, 1)
  for i in range(-k, k+1):
    discrete = np.append(discrete, gaussian(i, sigma, order))# * sigma**order)

  # The sum of the mask must be 1
  if order == 0:
    discrete = discrete / np.sum(discrete)

  # Calculate the gaussian function
  step = sizeMask / 100
  x_gauss = np.arange(-k, k+step, step)
  gauss = np.empty(0)
  for x in x_gauss:
    gauss = np.append(gauss, gaussian(x, sigma, order))

  # Scale the gaussian function to the discrete to compensate normalization
  gauss *= abs(max(discrete, key=abs) / max(gauss, key=abs))

  plt.bar(x_discrete, discrete, width=1.0, fill=False, label="Discrete")
  plt.plot(x_gauss, gauss, label="Gaussian")

  plt.rcParams["figure.figsize"] = (8,6)
  plt.legend()
  plt.title(title)

  plt.show()

def plotGauss(sigma, order, k=50, title=''):
  x = np.arange(-k, k+1, 1)
  y = []
  for i in x:
    y.append(gaussian(i, sigma, order))
  plt.rcParams["figure.figsize"] = (8,6)
  plt.plot(x, y, label='σ = '+str(sigma))
  plt.title(title)
  plt.legend()

def plotGaussSigma(sigma, order, k=50, title=''):
  x = np.arange(-k, k+1, 1)
  y = gaussianMask1D(sigma=sigma, sizeMask=2*k+1, order=order)
  plt.rcParams["figure.figsize"] = (8,6)
  plt.plot(x, y, label='σ = '+str(sigma))
  plt.title(title)
  plt.legend()

def displayMI_ES(vim, title='', factor=1):
  rows, columns = P0.imDistribution(len(vim))
  vim = vim.copy()

  # MODIFIED FROM P0: Normalize images
  for i in range(len(vim)):
    vim[i] = P0.rangeDisplay01(vim[i], 1)

  # First row
  out = vim[0].copy()
  for i in range(1, columns):
    out = np.hstack((out, vim[i]))

  # Adding more rows
  for i in range(1, rows):
    temp = vim[i * columns].copy()
    for j in range(1, columns):
      if i * columns + j < len(vim):
        temp = np.hstack((temp, vim[i * columns + j]))
      else:
        temp = np.hstack((temp, np.zeros(vim[i*columns].shape, dtype=vim[i*columns].dtype)))
    out = np.vstack((out, temp))

  return P0.displayIm(out, title, factor)

def my2DConv(im, sigma, orders):
  copy = im.copy()
  gaussianMask = gaussianMask1D(sigma=sigma, sizeMask=0, order=0).astype(dtype="float64")

  if orders == [0,0]:
    copy = cv2.sepFilter2D(src=copy, ddepth=cv2.CV_64F, kernelX=gaussianMask, kernelY=gaussianMask)

  if orders[0] != 0 and orders[1] == 0:
    maskX = gaussianMask1D(sigma=sigma, sizeMask=0, order=orders[0]).astype(dtype="float64")
    copy = cv2.sepFilter2D(src=copy, ddepth=cv2.CV_64F, kernelX=maskX, kernelY=gaussianMask)

  if orders[0] == 0 and orders[1] != 0:
    maskY = gaussianMask1D(sigma=sigma, sizeMask=0, order=orders[1]).astype(dtype="float64")
    copy = cv2.sepFilter2D(src=copy, ddepth=cv2.CV_64F, kernelX=gaussianMask, kernelY=maskY)

  if orders[0] != 0 and orders[1] != 0:
    maskX = gaussianMask1D(sigma=sigma, sizeMask=0, order=orders[0]).astype(dtype="float64")
    maskY = gaussianMask1D(sigma=sigma, sizeMask=0, order=orders[1]).astype(dtype="float64")
    im1 = cv2.sepFilter2D(src=copy.copy(), ddepth=cv2.CV_64F, kernelX=maskX, kernelY=gaussianMask)
    im2 = cv2.sepFilter2D(src=copy.copy(), ddepth=cv2.CV_64F, kernelX=gaussianMask, kernelY=maskY)
    copy = im1 + im2

  return copy

def gradientIM(im, sigma, debug=True):
  return my2DConv(im, sigma, [1,0]),  my2DConv(im, sigma, [0,1])

def laplacianG(im, sigma):
  return my2DConv(im, sigma, [2,2])

def colorOrient(im):
  output = np.zeros([im.shape[0], im.shape[1], 3])
  for i in range(im.shape[0]):
    for j in range(im.shape[1]):

      if im[i, j] >= -math.pi and im[i, j] < -math.pi/2:
        output[i, j] = np.array([255.0, 255.0, 0]) # Yellow
      elif  im[i, j] >= -math.pi/2 and im[i, j] < 0:
        output[i, j] = np.array([0, 255.0, 0]) # Green
      elif im[i, j] >= 0 and im[i, j] < math.pi/2:
        output[i, j] = np.array([0, 0, 255.0]) # Blue
      elif im[i, j] >= math.pi/2 and im[i, j] <= math.pi:
        output[i, j] = np.array([255.0, 0, 0]) # Red
      else:
        output[i, j] = np.array([0.0, 0.0, 0]) # Error

  return output

def pyramidGauss(im, sizeMask=7, nlevel=4):
  im = np.array(im, dtype=np.float64)
  vim = []
  vim.append(im.copy())
  gaussianMask = gaussianMask1D(sigma=0, sizeMask=sizeMask, order=0).astype(dtype="float64")

  for i in range(nlevel):
    # Remove half of the columns
    im = np.delete(im, list(range(0, im.shape[1], 2)), axis=1)
    # Remove half of the rows
    im = np.delete(im, list(range(0, im.shape[0], 2)), axis=0)
    # Apply Gaussian filter
    im = cv2.sepFilter2D(src=im, ddepth=cv2.CV_64F, kernelX=gaussianMask, kernelY=gaussianMask)
    vim.append(im.copy())

  return vim

def displayPyramid(vim, title=''):
  vim = vim.copy()
  # Normalize images
  for i in range(len(vim)):
    vim[i] = P0.rangeDisplay01(vim[i], 1)
    
  # Create an empty image
  if len(vim[i].shape) == 3:
    im = np.zeros((vim[0].shape[0], vim[0].shape[1] + vim[1].shape[1], 3))
  elif len(vim[i].shape) == 2:
    im = np.zeros((vim[0].shape[0], vim[0].shape[1] + vim[1].shape[1], 3))
  else:
    print("ERROR: Invalid image dimensions.")
    return 0

  # Add first image (biggest)
  for i in range(vim[0].shape[0]):
    for j in range(vim[0].shape[1]):
      im[i,j] = vim[0][i,j]

  # Add other images
  posX = 0
  posY = vim[0].shape[1]
  for n in range(1, len(vim)):
    for i in range(vim[n].shape[0]):
      for j in range(vim[n].shape[1]):
        im[i + posX, j + posY] = vim[n][i,j]
    posX += vim[n].shape[0]

  return P0.displayIm(im, title=title, factor=1.5)

def pyramidLap(im, sizeMask, nlevel=4, flagInterp=cv2.INTER_LINEAR):
  im = np.array(im, dtype=np.float64)
  # Gaussian pyramid
  vim = pyramidGauss(im, sizeMask, nlevel)
  vimL = []

  for i in range(1, len(vim)):
    # Calculate difference between an image and itself smoothed
    resize = cv2.resize(vim[i], dsize=(vim[i-1].shape[1], vim[i-1].shape[0]), interpolation=flagInterp)
    im = vim[i-1] - resize
    vimL.append(im)

  vimL.append(vim[len(vim)-1])
  return vimL

def reconstructIm(pyL, flagInterp=cv2.INTER_LINEAR):
  # Start with the smoothed image
  im = pyL[len(pyL)-1]
  for i in range(len(pyL)-2, -1, -1):
    # Resize to the next image
    im = cv2.resize(im, pyL[i].T.shape, interpolation=flagInterp)
    # Add the difference
    im += pyL[i]
  return im

def displayReconstructIm(pyL, flagInterp=cv2.INTER_LINEAR):
  # Start with the smoothed image
  im = pyL[len(pyL)-1]
  vim = []
  vim.append(im.copy())
  for i in range(len(pyL)-2, -1, -1):
    # Resize to the next image
    im = cv2.resize(im, pyL[i].T.shape, interpolation=flagInterp)
    # Add the difference
    im += pyL[i]
    vim.append(im.copy())
  return displayPyramid(vim[::-1], 'Reconstructed Images')

def bonus(vim, sigma_low, sigma_high):
  low = my2DConv(im=vim[0], sigma=sigma_low, orders=[0,0])
  # Option 1:
  # high = pyramidLap(im=vim[1], sizeMask=7, nlevel=4)[0]
  # Option 2:
  high = vim[1] - my2DConv(vim[1], sigma_high, [0,0])
  im = low + high
  pyrG = pyramidGauss(im=im, sizeMask=7, nlevel=4)
  return displayPyramid(pyrG, title="Bonus")
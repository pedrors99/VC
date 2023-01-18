import math
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def readIm(filename, flagColor=1):
  # Read image
  image = cv2.imread(filename, flagColor)

  # Swap BGR to RGB
  if flagColor == 1:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  return image

def rangeDisplay01(im, flag_GLOBAL):
  # Check image type (grayscale or color)
  if len(im.shape) == 2 or flag_GLOBAL == 1:
    # Normalize the grayscale image or globally the RGB image
    # Compute range and apply normalization
    max = im.max()
    min = im.min()
    im = (im - min)/(max - min)

  else:
    # Normalize each band as a grayscale image
    norm = np.zeros(im.shape, dtype=float)
    min = np.min(im, axis=(0, 1))
    max = np.max(im, axis=(0, 1))
    im = (im[:, :] - min) / (max - min)

  return im

def displayIm(im, title='Result', factor=1, showFlag=np.True_):
  # Normalize range
  im = rangeDisplay01(im, 1)

  fig, ax = plt.subplots()
  # Display the image
  if len(im.shape) == 3:
    # im has three channels (RGB)
    ax.imshow(im)
  else:
    # im has a single channel (Grayscale)
    ax.imshow(im, cmap='gray')
    
  figure_size = plt.gcf().get_size_inches()
  plt.title(title) # Adding title
  plt.gcf().set_size_inches(factor * figure_size)
  plt.xticks([]), plt.yticks([]) # Axis label off
  if showFlag: plt.show()

def imDistribution(size):
  # Calculates the number of rows and columns for the "best" distribution of
  # multiple images using the square root of the number of images
  root = math.sqrt(size)
  rows = root
  columns = root
  if root % 1 != 0:
      columns += 1
      if root % 1 > 0.5:
        rows += 1
  return int(rows), int(columns)

def displayMI_ES(vim, title='', factor=1):
  # Let's start with case (a). We concatenate the images by columns, or by rows 
  # and columns, depending on the number of images and their dimensions
  rows, columns = imDistribution(len(vim))

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

  return displayIm(out, title, factor)

def fillImage(im_, rows, columns):
  im = im_.copy()
  
  # Adding empty rows
  if im.shape[0] < rows:
    im = np.vstack((im, np.zeros((rows - im.shape[0], im.shape[1], im.shape[2]))))
    
  # Adding empty columns  
  if im.shape[1] < columns:
    im = np.hstack((im, np.zeros((im.shape[0], columns - im.shape[1], im.shape[2]))))
    
  return im

def displayMI_NES(vim):
  # For arrays containing gray and RGB images, we convert all in RGB images
  for i in range(len(vim)):
    if len(vim[i].shape) == 2:
      vim[i] = cv2.cvtColor(vim[i],cv2.COLOR_GRAY2RGB)

  rows, columns = imDistribution(len(vim))

  # Max size of the images in each row
  rowSize = []
  for i in range(rows):
    max = 0
    for j in range(columns):
      if int(i * columns + j) < len(vim) and vim[int(i * columns + j)].shape[0] > max:
        max = vim[int(i * columns + j)].shape[0]
    rowSize.append(max)

  # Max size of the images in each column
  columnSize = []
  for j in range(columns):
    max = 0
    for i in range(rows):
      if int(i * columns + j) < len(vim) and vim[int(i * columns + j)].shape[1] > max:
        max = vim[int(i * columns + j)].shape[1]
    columnSize.append(max)
    
  # Extend all the images to the size of the largest images of their row and column
  extendedVim = []
  for i in range(rows):
    for j in range(columns):
      extendedVim.append(fillImage(vim[int(i * columns + j)], rowSize[i], columnSize[j]))

  # Almost the same code as in displayMI_ES()
  # First row
  out = extendedVim[0].copy()
  for i in range(1, columns):
    out = np.hstack((out, extendedVim[i]))

  # Adding more rows
  for i in range(1, rows):
    temp = extendedVim[i * columns].copy()
    for j in range(1, columns):
      if i * columns + j < len(extendedVim):
        temp = np.hstack((temp, extendedVim[i * columns + j]))
      else:
        temp = np.hstack((temp, np.zeros(extendedVim[i*columns].shape, dtype=extendedVim[i*columns].dtype)))
    out = np.vstack((out, temp))
    
  return out

def changePixelValues(im, cp, nv):
  # cp is a vector of pixel coordinates
  # nv is a vector with the new values
  # replace the values of cp with the nv values
  for i in range(len(cp)):
    im[cp[i,0], cp[i,1]] = nv[i]
  return displayIm(im)

def print_images_titles(vim, titles=None, rows=0):
  fig = plt.figure()
  i=1

  if rows > 0:
    # Funcion defined in exercise 3 for the 'best' distribution
    columns = math.ceil(len(vim)/rows)
  else:
    rows, columns = imDistribution(len(vim))

  # Add images
  for image in vim:
    ax = fig.add_subplot(rows, columns, i)
    if titles is not None:
      ax.title.set_text(titles[i-1])
    ax.imshow(image)
    ax.axis('off') # Remove axis
    i+=1
  fig.tight_layout()
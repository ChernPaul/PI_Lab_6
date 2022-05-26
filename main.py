import numpy as np
from skimage.io import imsave, imshow, show, imread
from matplotlib import pyplot as plt
import math
from scipy.stats import entropy

PATH_TO_FILE = "D:\\Рабочий стол\\Test Images\\14_LENA.TIF"


def MyDifCode(x, e, r):  # x - source image  e - max error rate  r - predictor number
    size = list(np.shape(x))
    size[1] = size[1] + 1
    size = tuple(size)
    y = np.zeros(size)  # recovered signal
    p = np.zeros(size)  # predicted signal
    f = np.zeros(size)  # difference signal
    q = np.zeros(size)  # quantized difference signal
    for i in range(0, len(x), 1):
        for j in range(0, len(x[0]), 1):
            if r == 1:
                p[i][j] = y[i][j - 1]
            if r == 2:
                p[i][j] = 0.5 * (y[i][j - 1] + y[i - 1][j])
            if r == 3:
                p[i][j] = 0.25 * (y[i][j - 1] + y[i - 1][j] + y[i - 1][j - 1] + y[i - 1][j + 1])
            if r == 4:
                p[i][j] = y[i][j - 1] + y[i - 1][j] - y[i - 1][j - 1]
            f[i][j] = x[i][j] - p[i][j]
            q[i][j] = np.sign(f[i][j]) * math.floor((np.abs(f[i][j]) + e)/(2*e + 1))  #  round to down
            y[i][j] = p[i][j] + q[i][j] * (2*e + 1)
    if e == 0:  # task 6 show difference signal for epsilon = 0
        fig = plt.figure(figsize=(20, 10))
        fig.add_subplot(1, 1, 1)
        plt.title(f"f(dif_img) with predictor {r}")
        imshow(f, cmap='gray')  # , vmin=0, vmax=255  autocontrast
        show()
    return q  # massive quantized differences


def MyDifDecode(q, e, r):  # q -  massive quantized differences  e - max error rate  r - predictor number
    size = np.shape(q)
    y = np.zeros(size)   # recovered signal
    p = np.zeros(size)   # predicted signal
    for i in range(0, len(q), 1):
        for j in range(0, len(q[0]) - 1, 1):
            if r == 1:
                p[i][j] = y[i][j - 1]
            if r == 2:
                p[i][j] = 0.5 * (y[i][j - 1] + y[i - 1][j])
            if r == 3:
                if j + 1 == len(x[0]):
                    y[i - 1][j + 1] = 0
                p[i][j] = 0.25 * (y[i][j - 1] + y[i - 1][j] + y[i - 1][j - 1] + y[i - 1][j + 1])
            if r == 4:
                p[i][j] = y[i][j - 1] + y[i - 1][j] - y[i - 1][j - 1]
            y[i][j] = p[i][j] + q[i][j] * (2 * e + 1)
    return y[:, :-1]  # y - decompressed image  cut last column


def process_entropy_of_image(q):
    hist, bins = np.histogram(q, bins=[el for el in range(int(q.min()), int(q.max() + 1), 1)])
    W = hist / q.size
    entrpy = entropy(W, base=2)
    return entrpy


x = imread(PATH_TO_FILE)
array_of_epsilons = []

entropy_array1 = []
entropy_array2 = []
entropy_array3 = []
entropy_array4 = []

max_epsilon1 = []
max_epsilon2 = []
max_epsilon3 = []
max_epsilon4 = []

q1_array = []
q2_array = []
q3_array = []
q4_array = []

img_decompress1 = []
img_decompress2 = []
img_decompress3 = []
img_decompress4 = []

for e in range(0, 51, 1):
    q1 = MyDifCode(x, e, 1)
    q2 = MyDifCode(x, e, 2)
    q3 = MyDifCode(x, e, 3)
    q4 = MyDifCode(x, e, 4)
    if (e == 0) or (e == 5) or (e == 10):  # task 7 show compressed images for epsilon = 0, 5, 10
        q1_array.append(q1)
        q2_array.append(q2)
        q3_array.append(q3)
        q4_array.append(q4)
    y1 = MyDifDecode(q1, e, 1)
    y2 = MyDifDecode(q2, e, 2)
    y3 = MyDifDecode(q3, e, 3)
    y4 = MyDifDecode(q4, e, 4)
    if (e == 5) or (e == 10) or (e == 20) or (e == 40):  # task 5 show decompressed images for epsilon = 5, 10,20 , 40
        img_decompress1.append(y1)
        img_decompress2.append(y2)
        img_decompress3.append(y3)
        img_decompress4.append(y4)

    array_of_epsilons.append(e)
    entropy_array1.append(process_entropy_of_image(q1))
    entropy_array2.append(process_entropy_of_image(q2))
    entropy_array3.append(process_entropy_of_image(q3))
    entropy_array4.append(process_entropy_of_image(q4))

    max_epsilon1.append(np.max(x - y1))  # for checking error rate in console  task 8
    max_epsilon2.append(np.max(x - y2))
    max_epsilon3.append(np.max(x - y3))
    max_epsilon4.append(np.max(x - y4))
    print(e)  # show current state optional

for i in range(0, len(max_epsilon1), 1):
    print(f"e: {array_of_epsilons[i]}\tmax1: {max_epsilon1[i]}\t max2: {max_epsilon2[i]:.5f}\t max3: {max_epsilon3[i]:.5f}\t max4: {max_epsilon4[i]}")

fig = plt.figure(figsize=(20, 10))
for i in range(0, len(img_decompress1), 1):
    fig.add_subplot(4, 4, i + 1)
    plt.title(f"Decompressed image with e:{(2 ** i) * 5}")
    imshow(img_decompress1[i], cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(4, 4, i + 5)
    plt.title(f"Decompressed image with e:{(2 ** i) * 5}")
    imshow(img_decompress2[i], cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(4, 4, i + 9)
    plt.title(f"Decompressed image with e:{(2 ** i) * 5}")
    imshow(img_decompress3[i], cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(4, 4, i + 13)
    plt.title(f"Decompressed image with e:{(2 ** i) * 5}")
    imshow(img_decompress4[i], cmap='gray', vmin=0, vmax=255)
fig.savefig('Decompressed_images.jpg')
show()

fig1 = plt.figure(figsize=(20, 10))
for i in range(0, len(q1_array), 1):
    fig1.add_subplot(4, 3, i + 1)
    plt.title(f"Compressed image with e:{i * 5}")
    imshow(q1_array[i], cmap='gray')  # , vmin=0, vmax=255
    fig1.add_subplot(4, 3, i + 4)
    plt.title(f"Compressed image with e:{i * 5}")
    imshow(q2_array[i], cmap='gray')  # , vmin=0, vmax=255
    fig1.add_subplot(4, 3, i + 7)
    plt.title(f"Compressed image with e:{i * 5}")
    imshow(q3_array[i], cmap='gray')  # , vmin=0, vmax=255
    fig1.add_subplot(4, 3, i + 10)
    plt.title(f"Compressed image with e:{i * 5}")
    imshow(q4_array[i], cmap='gray')  # , vmin=0, vmax=255
fig1.savefig('Compressed_images.jpg')
show()

fig2 = plt.figure(figsize=(20, 10))
fig2.add_subplot(1, 1, 1)
plt.plot(array_of_epsilons, entropy_array1, color='red', linestyle='-', linewidth=1)
plt.plot(array_of_epsilons, entropy_array2, color='blue', linestyle='-', linewidth=1)
plt.plot(array_of_epsilons, entropy_array3, color='green', linestyle='-', linewidth=1)
plt.plot(array_of_epsilons, entropy_array4, color='yellow', linestyle='-', linewidth=1)
plt.legend(['predictor 1', 'predictor 2', 'predictor 3', 'predictor 4'])
fig2.savefig('Entropy_value_from_e.jpg')
show()

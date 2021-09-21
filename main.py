import queue
import cv2
import numpy as np

colorsList = []
codesList = []
sumIndex = 0

#truncColorList = [0]*100


class Node:
    def __init__(self):
        self.prob = None
        self.code = None
        self.data = None
        self.left = None
        self.right = None  # the color (the bin value) is only required in the leaves

    def __lt__(self, other):
        if self.prob < other.prob:  # define rich comparison methods for sorting in the priority queue
            return 1
        else:
            return 0

    def __ge__(self, other):
        if self.prob > other.prob:
            return 1
        else:
            return 0


def tree(probabilities):
    prq = queue.PriorityQueue()
    for color, probability in enumerate(probabilities):
        leaf = Node()
        leaf.data = color
        leaf.prob = probability
        prq.put(leaf)

    while prq.qsize() > 1:
        newnode = Node()  # create new node
        l = prq.get()
        r = prq.get()  # get the smallest probs in the leaves
        # remove the smallest two leaves
        newnode.left = l  # left is smaller
        newnode.right = r
        newprob = l.prob + r.prob  # the new prob in the new node must be the sum of the other two
        newnode.prob = newprob
        prq.put(newnode)  # new node is inserted as a leaf, replacing the other two
    return prq.get()  # return the root node - tree is complete


def huffman_traversal(root_node, tmp_array, f, k,truncColorList):  # traversal of the tree to generate codes
    global colorsList
    global codesList
    #global truncColorList
    global sumIndex
    if root_node.left is not None:
        tmp_array[huffman_traversal.count] = 1
        huffman_traversal.count += 1
        huffman_traversal(root_node.left, tmp_array, f, k,truncColorList)
        huffman_traversal.count -= 1
    if root_node.right is not None:
        tmp_array[huffman_traversal.count] = 0
        huffman_traversal.count += 1
        huffman_traversal(root_node.right, tmp_array, f, k,truncColorList)
        huffman_traversal.count -= 1
    else:
        # count the number of bits for each color
        huffman_traversal.output_bits[root_node.data] = huffman_traversal.count
        bitstream = ''.join(str(cell) for cell in tmp_array[1:huffman_traversal.count])
        c = 0
        color = str(root_node.data)



        if color == str(sumIndex):
            i = 0
            while i < k:
                colorsList += [truncColorList[i]]
                newBitStream = str(bitstream) + str(bin(i))
                codesList = codesList + [newBitStream]
                i += 1
                c += 1
        else:
            colorsList = colorsList + [c + int(color)]
            codesList = codesList + [bitstream]

        wr_str = color + ' ' + bitstream + '\n'
        f.write(wr_str)  # write the color and the code to a file
    return


# def getMinNoOfKs(histogram, k):
#     global truncColorList
#     h = np.sort(histogram)
#     i = 1
#     while i <= k:
#         truncColorList += h[i]
#         i += 1


def truncate(histogram, k, imgH, imgW, truncColorList):
    global sumIndex
    global colorsList
    global codesList
    #global truncColorList

    probabilities = histogram / (imgH * imgW)  # calculating the probabilities using histogram frequencies
    sortedProb = np.sort(probabilities)
    finalProArr = sortedProb[k:]

    truncSum = np.sum(truncColorList)  # calculate their sum

    print("Before adding sum: ", len(finalProArr))
    out = np.insert(finalProArr, 0, truncSum)  # Insert the Sum value in our array
    out = np.sort(out)
    sumIndex = out.tolist().index(truncSum)  # Get the index of Sum after inserting it in our array
    print("Sum at index: ", sumIndex)
    print(out)
    print("Sum is: ", truncSum)
    print(type(finalProArr))
    print("After adding sum: ", len(out))
    return out


if __name__ == '__main__':
    # Read an bmp image into a numpy array
    img = cv2.imread("cameraman.tif", 0)
    h = img.shape[0]
    w = img.shape[1]

    # compute histogram of pixels
    a = np.bincount(img.ravel(), minlength=256)
    b = np.array([0])
    hist = np.setdiff1d(a, b, True)

    #truncColorList = [0] * 100
    h2 = hist
    truncColorList = []

    i = 0
    while i < 3:
        truncColorList += [h2.tolist().index(np.min(h2))]
        h2[h2.tolist().index(np.min(h2))] = 10000000  # Any large number so we can get rid of this minimum and get the next min
        i += 1

    trancArr = truncate(hist, 3, h, w, truncColorList)

    root_node = tree(trancArr)  # create the tree using the probs.
    tmp_array = np.ones([64], dtype=int)
    huffman_traversal.output_bits = np.empty(len(hist), dtype=int)
    huffman_traversal.count = 0
    f = open('Dict.txt', 'w')
    # f.write(str(len(sumTruncArr + probabilities[k:])) + '\n')

    huffman_traversal(root_node, tmp_array, f, 3, truncColorList)  # traverse the tree and write the codes
    print("Colors: ", colorsList)
    print("Code: ", codesList)
    f.close()

    # print(len(sumTruncArr))
    f = open('BinCode.txt', 'w')
    for x in range(h):
        for y in range(w):
            s = str(codesList[colorsList.index(img[x][y])])
            f.write(s)
    f.close()

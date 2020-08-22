import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def extract_biggest_bounding_box(contours):
    biggest = None
    w = 0
    h = 0
    for cnt in contours:
        _, _, cw, ch = cv.boundingRect(cnt)
        if cw * ch > w * h:
            w, h = cw, ch
            biggest = cnt
    return biggest


def point_inside_square(x1, y1, x2, y2, x3, y3):
    return x3 >= x1 and x3 <= x2 and y3 >= y1 and y3 <= y2


def bounding_inside(cnt1, cnt2):
    x1, y1, w1, h1 = cv.boundingRect(cnt1)
    x2, y2, w2, h2 = cv.boundingRect(cnt2)
    return point_inside_square(x1, y1, x1 + w1, y1 + h1, x2, y2) and point_inside_square(x1, y1, x1 + w1, y1 + h1, x2 + w2, y2 + h2)


def extract_board(contours):
    ret = []
    biggest = extract_biggest_bounding_box(contours)
    for cnt in contours:
        if bounding_inside(biggest, cnt):
            ret.append(cnt)
    return ret


def extract_board_properties(contours):
    circles = []
    cells = []

    # extract circles : boxes that contains only one box inside
    for cnt in contours:
        inside = 0
        for cnt2 in contours:
            if bounding_inside(cnt, cnt2):
                inside = inside + 1
        if inside == 2:
            circles.append(cnt)

    # extract cells : boxes that aren't circles or inside a circle
    for cnt in contours:
        inside = 0
        for cnt2 in contours:
            if bounding_inside(cnt2, cnt):
                inside += 1
        if inside == 2:
            cells.append(cnt)

    return cells, circles


def extract_cols(contours):
    thresh = 10
    ret = {}
    for cnt in contours:
        x, y, _, _, = cv.boundingRect(cnt)
        if x - x % thresh in ret:
            ret[x - x % thresh][y] = cnt
        else:
            ret[x - x % thresh] = {y: cnt}
    return ret


def extract_grid(img, cells, circles):
    # extract cols
    cols = extract_cols(cells)

    # compute #rows, #cols
    n = len(cols[list(cols.keys())[0]])
    m = len(cols)

    # init grid
    grid = []
    grid_cnt = []
    for _ in range(n):
        grid.append([])
        grid_cnt.append([])
        for _ in range(m):
            grid[-1].append(0)
            grid_cnt[-1].append(None)

    j = 0
    for col in sorted(cols.keys()):
        i = 0
        for row in sorted(cols[col].keys()):
            grid_cnt[i][j] = cols[col][row]
            i = i + 1
        j = j + 1

    # detect cells containing circles
    colors_mapping = {}
    for i in range(n):
        for j in range(m):
            for cnt in circles:
                if bounding_inside(grid_cnt[i][j], cnt):
                    x, y, w, h = cv.boundingRect(grid_cnt[i][j])
                    color = repr(img[y + h // 2, x + w // 2])
                    if (color in colors_mapping) == False:
                        colors_mapping[color] = len(colors_mapping) + 1
                    grid[i][j] = colors_mapping[color]
                    break
    # printing
    for row in grid:
        for col in row:
            print(col, end='')
        print()

    return n, m, grid


# read image
img = cv.imread('example.jpg')
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# edge detection
edges = cv.Canny(gray, 50, 150)
kernel = np.ones((5, 5), np.uint8)
edges = cv.dilate(edges, kernel, iterations=1)

# contours detection
contours, hierarchy = cv.findContours(
    edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# extract board
contours = extract_board(contours)
cells, circles = extract_board_properties(contours)

# drawing cells
# for cnt in cells:
#     x, y, w, h = cv.boundingRect(cnt)
#     cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# drawing circles
# for cnt in circles:
#     x, y, w, h = cv.boundingRect(cnt)
#     cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# construct grid
n, m, grid = extract_grid(img, cells, circles)

# ploting
# plt.imshow(img)
# plt.show()

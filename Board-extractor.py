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
        _, _, w, h = cv.boundingRect(cnt)
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
            if bounding_inside(cnt2, cnt):
                inside = inside + 1
        if inside == 3:
            circles.append(cnt)
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
    inv_colors_mapping = {}
    for i in range(n):
        for j in range(m):
            for cnt in circles:
                if bounding_inside(grid_cnt[i][j], cnt):
                    x, y, w, h = cv.boundingRect(grid_cnt[i][j])
                    color = repr(img[y + h // 2, x + w // 2])
                    if (color in colors_mapping) == False:
                        colors_mapping[color] = len(colors_mapping) + 1
                        inv_colors_mapping[colors_mapping[color]
                                           ] = img[y + h // 2, x + w // 2]
                    grid[i][j] = colors_mapping[color]
                    break
    # printing
    # for row in grid:
    #     for col in row:
    #         print(col, end='')
    #     print()
    # print("###########")
    return n, m, grid, grid_cnt, inv_colors_mapping


def color_image(img, grid, grid_cnt, inv_colors_mapping):
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            x, y, w, h = cv.boundingRect(grid_cnt[i][j])
            # print(' ' + inv_colors_mapping[grid[i][j]] + ' ', end='')
            color = (int(inv_colors_mapping[grid[i][j]][0]),
                     int(inv_colors_mapping[grid[i][j]][1]), int(inv_colors_mapping[grid[i][j]][2]))
            cv.rectangle(img, (x, y), (x+w, y+h), color, -1)


dx = [1, -1, 0, 0]
dy = [0, 0, 1, -1]


def is_valid(x, y, grid):
    return x >= 0 and x < len(grid) and y >= 0 and y < len(grid[0])


def dfs(x, y, stx, sty, grid, color):
    for d in range(4):
        nwx = x + dx[d]
        nwy = y + dy[d]
        if is_valid(nwx, nwy, grid):
            if grid[nwx][nwy] == color and (nwx != stx or nwy != sty) and solve_grid(grid, color + 1):
                return True
            if grid[nwx][nwy] == 0:
                grid[nwx][nwy] = -color
                if dfs(nwx, nwy, stx, sty, grid, color):
                    return True
                grid[nwx][nwy] = 0
    return False


def solve_grid(grid, color):
    stx = -1
    sty = -1
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if(grid[i][j] == color):
                stx = i
                sty = j
                break
    if stx == -1:
        # printing
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                grid[i][j] = abs(grid[i][j])
        return grid
    return dfs(stx, sty, stx, sty, grid, color)


# read image
img_path = 'example6.jpg'
img = cv.imread(img_path)
original = cv.imread(img_path)
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# edge detection
edges = cv.Canny(gray, 100, 150)
kernel = np.ones((5, 5), np.uint8)
edges = cv.dilate(edges, kernel, iterations=1)

# contours detection
contours, hierarchy = cv.findContours(
    edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# extract board
contours = extract_board(contours)
cells, circles = extract_board_properties(contours)

# draw board
# for cnt in contours:
#     x, y, w, h = cv.boundingRect(cnt)
#     cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# drawing cells
# for cnt in cells:
#     x, y, w, h = cv.boundingRect(cnt)
#     cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# drawing circles
# for cnt in circles:
#     x, y, w, h = cv.boundingRect(cnt)
#     cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# construct grid
n, m, grid, grid_cnt, inv_colors_mapping = extract_grid(img, cells, circles)

solve_grid(grid, 1)
color_image(img, grid, grid_cnt, inv_colors_mapping)

# ploting
f, axarr = plt.subplots(1, 2)
axarr[0].imshow(original)
axarr[1].imshow(img)
plt.show()

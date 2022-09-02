def auto(self):
    self.baseimg = gdalnumeric.LoadFile(self.base)
    self.wrapimg = gdalnumeric.LoadFile(self.wrap)
    dataSet = []
    bas = []
    wra = []
    with open(self.GCP) as fr:
        for line in fr.readlines()[5:8]:
            curline = line.strip()
            test = curline.split()
            x1, y1 = float(test[0]), float(test[1])
            x2, y2 = float(test[2]), float(test[3])
            bas.append((x1, y1))
            wra.append((x2, y2))
            dataSet.append(test)
    wrap = np.ones((3, 3))
    base = np.ones((3, 2))
    for i in range(3):
        wrap[i][1], wrap[i][2] = wra[i]
        base[i][0], base[i][1] = bas[i]

    a1 = np.dot(wrap.transpose(), wrap)
    B = np.linalg.inv(a1)
    coeff = np.dot(np.dot(B, wrap.transpose()), base)  # wrap:  (6, 498, 414) to  base:  (1390, 1071)
    # print(coeff)

    a, b = self.wrapimg.shape[1], self.wrapimg.shape[2]
    test = np.zeros((4, 3))
    test[0][0] = 1
    test[1][0] = 1
    test[2][0] = 1
    test[3][0] = 1
    test[1][2] = b
    test[2][1] = a
    test[3][1] = a
    test[3][2] = b
    res = np.dot(test, coeff)
    x_range = max(res[:, 0]) - min(res[:, 0])  # new image
    y_range = max(res[:, 1]) - min(res[:, 1])

    '''

    wrap = np.ones((3, 2))
    base = np.ones((3, 3))
    for i in range(3):
        wrap[i][0], wrap[i][1] = wra[i]
        base[i][1], base[i][2] = bas[i]

    ra = np.dot(base.transpose(), base)
    rb = np.linalg.inv(ra)
    rcoe = np.dot(np.dot(rb, base.transpose()), wrap)             #base to wrap

    delta_x = abs(min(res[:, 0]))                               #delta of ordinate
    delta_y = abs(min(res[:, 1]))
    xy_list = [(1, i, j) for i in range(int(x_range)) for j in range(int(y_range))]
    xy = np.array(xy_list)
    xy[:, 1] = xy[:, 1] - delta_x
    xy[:, 2] = xy[:, 2] - delta_y

    wrap_pix_loc = np.dot(xy, rcoe)    #1594*1409  *  2

    f = open("check.txt", "w")
    f.write(str(wrap_pix_loc[:1000]))
    f.close()

    newimg = np.zeros((int(x_range)*int(y_range), self.wrapimg.shape[0]))    #6*1594*1409   2245946*6
    self.wrapimg = self.wrapimg.reshape((self.wrapimg.shape[1]*self.wrapimg.shape[2], self.wrapimg.shape[0]))     #206172*6
    for i in range(xy.shape[0]):
        if wrap_pix_loc[i][0]<0 or wrap_pix_loc[i][1]<0 or wrap_pix_loc[i][0]>self.wrapimg.shape[0] or wrap_pix_loc[i][1]>self.wrapimg.shape[1]:
            pass
        else:
            #print(round(wrap_pix_loc[i][0]*wrap_pix_loc[i][1]))
            newimg[i, :] = self.wrapimg[ int(round(wrap_pix_loc[i][0]*wrap_pix_loc[i][1])) , :]

    R = newimg[:, 0].reshape((int(x_range), int(y_range)))
    G = newimg[:, 1].reshape((int(x_range), int(y_range)))
    B = newimg[:, 2].reshape((int(x_range), int(y_range)))
    self.img = cv2.merge([R, G, B])
    cv2.imwrite(self.savepath, self.img)
    print(1)
    '''

    p1 = compute.Point(res[0][0], res[0][1])
    p2 = compute.Point(res[1][0], res[1][1])
    p3 = compute.Point(res[2][0], res[2][1])
    p4 = compute.Point(res[3][0], res[3][1])
    fx = (compute.Getlen(p1, p3).getlen() + compute.Getlen(p2, p4).getlen()) / 2
    fy = (compute.Getlen(p1, p2).getlen() + compute.Getlen(p3, p4).getlen()) / 2

    self.img = np.transpose(self.wrapimg, [1, 2, 0]).astype(self.wrapimg.dtype)

    # angel
    ang = (res[0][1] - test[0][2]) / (res[0][0] - test[0][1])
    angel = math.atan(ang) * 180 / (math.pi)
    (h, w) = (int(x_range), int(y_range))
    # center = np.dot([1, int(self.wrapimg.shape[1] / 2), int(self.wrapimg.shape[2] / 2)], coeff).astype(self.wrapimg.dtype)
    center = (int(self.img.shape[0] / 2), int(self.img.shape[1] / 2))

    # M = cv2.getRotationMatrix2D((center[0], center[1]), -angel, 1.0)
    M = cv2.getRotationMatrix2D(center, -angel, 1.0)
    rotated = cv2.warpAffine(self.img[:, :, :3], M, (self.img.shape[1] * 2, self.img.shape[0]))
    pic = cv2.resize(rotated, (int(fx), int(fy)))
    self.img = pic
    cv2.imwrite(self.savepath, self.img)

    # self.img = pic
    # cv2.imwrite(self.savepath, self.img)

    '''
    base = np.ones((3, 3))
    wrap = np.ones((3, 2))
    for i in range(3):
        base[i][1], base[i][2] = bas[i]
        wrap[i][0], wrap[i][1] = wra[i]
    A_abs = np.linalg.det(base)
    B = np.linalg.inv(base)
    base_ = B / A_abs
    coeff = np.dot(np.dot(np.dot(base.transpose(), base), base_), wrap)
    a1 = np.dot(base.transpose(), base)
    B = np.linalg.inv(a1)
    coeff = np.dot(np.dot(B, base.transpose()), wrap)

    a, b = self.baseimg.shape[0], self.baseimg.shape[1]
    test = np.zeros((4, 3))
    test[0][0] = 1
    test[1][0] = 1
    test[2][0] = 1
    test[3][0] = 1
    test[1][2] = b
    test[2][1] = a
    test[3][1] = a
    test[3][2] = b

    res = np.dot(test, coeff)
    #print(res)
    x_range = max(res[:, 0]) - min(res[:, 0])
    y_range = max(res[:, 1]) - min(res[:, 1])
    #print(math.ceil(x_range), math.ceil(y_range))

    newimg = np.zeros((self.wrapimg.shape[0], math.ceil(x_range), math.ceil(y_range)))
    '''
    return



def auto(self):
    self.baseimg = gdalnumeric.LoadFile(self.base)
    self.wrapimg = gdalnumeric.LoadFile(self.wrap)
    dataSet = []
    bas = []
    wra = []
    with open(self.GCP) as fr:
        for line in fr.readlines()[5:8]:
            curline = line.strip()
            test = curline.split()
            x1, y1 = float(test[0]), float(test[1])
            x2, y2 = float(test[2]), float(test[3])
            bas.append((x1, y1))
            wra.append((x2, y2))
            dataSet.append(test)
    wrap = np.ones((3, 3))
    base = np.ones((3, 2))
    for i in range(3):
        wrap[i][1], wrap[i][2] = wra[i]
        base[i][0], base[i][1] = bas[i]

    a1 = np.dot(wrap.transpose(), wrap)
    B = np.linalg.inv(a1)
    coeff = np.dot(np.dot(B, wrap.transpose()), base)  # wrap:  (6, 498, 414) to  base:  (1390, 1071)
    # print(coeff)

    a, b = self.wrapimg.shape[1], self.wrapimg.shape[2]
    test = np.zeros((4, 3))
    test[0][0] = 1
    test[1][0] = 1
    test[2][0] = 1
    test[3][0] = 1
    test[1][2] = b
    test[2][1] = a
    test[3][1] = a
    test[3][2] = b
    res = np.dot(test, coeff)
    x_range = max(res[:, 0]) - min(res[:, 0])  # new image
    y_range = max(res[:, 1]) - min(res[:, 1])

    '''

    wrap = np.ones((3, 2))
    base = np.ones((3, 3))
    for i in range(3):
        wrap[i][0], wrap[i][1] = wra[i]
        base[i][1], base[i][2] = bas[i]

    ra = np.dot(base.transpose(), base)
    rb = np.linalg.inv(ra)
    rcoe = np.dot(np.dot(rb, base.transpose()), wrap)             #base to wrap

    delta_x = abs(min(res[:, 0]))                               #delta of ordinate
    delta_y = abs(min(res[:, 1]))
    xy_list = [(1, i, j) for i in range(int(x_range)) for j in range(int(y_range))]
    xy = np.array(xy_list)
    xy[:, 1] = xy[:, 1] - delta_x
    xy[:, 2] = xy[:, 2] - delta_y

    wrap_pix_loc = np.dot(xy, rcoe)    #1594*1409  *  2

    f = open("check.txt", "w")
    f.write(str(wrap_pix_loc[:1000]))
    f.close()

    newimg = np.zeros((int(x_range)*int(y_range), self.wrapimg.shape[0]))    #6*1594*1409   2245946*6
    self.wrapimg = self.wrapimg.reshape((self.wrapimg.shape[1]*self.wrapimg.shape[2], self.wrapimg.shape[0]))     #206172*6
    for i in range(xy.shape[0]):
        if wrap_pix_loc[i][0]<0 or wrap_pix_loc[i][1]<0 or wrap_pix_loc[i][0]>self.wrapimg.shape[0] or wrap_pix_loc[i][1]>self.wrapimg.shape[1]:
            pass
        else:
            #print(round(wrap_pix_loc[i][0]*wrap_pix_loc[i][1]))
            newimg[i, :] = self.wrapimg[ int(round(wrap_pix_loc[i][0]*wrap_pix_loc[i][1])) , :]

    R = newimg[:, 0].reshape((int(x_range), int(y_range)))
    G = newimg[:, 1].reshape((int(x_range), int(y_range)))
    B = newimg[:, 2].reshape((int(x_range), int(y_range)))
    self.img = cv2.merge([R, G, B])
    cv2.imwrite(self.savepath, self.img)
    print(1)
    '''

    p1 = compute.Point(res[0][0], res[0][1])
    p2 = compute.Point(res[1][0], res[1][1])
    p3 = compute.Point(res[2][0], res[2][1])
    p4 = compute.Point(res[3][0], res[3][1])
    fx = (compute.Getlen(p1, p3).getlen() + compute.Getlen(p2, p4).getlen()) / 2
    fy = (compute.Getlen(p1, p2).getlen() + compute.Getlen(p3, p4).getlen()) / 2

    self.img = np.transpose(self.wrapimg, [1, 2, 0]).astype(self.wrapimg.dtype)
    pic = cv2.resize(self.img[:, :, :3], (int(fx), int(fy)))

    # angel
    ang = (res[0][1] - test[0][2]) / (res[0][0] - test[0][1])
    angel = math.atan(ang) * 180 / (math.pi)
    (h, w) = (int(x_range), int(y_range))
    center = (int(x_range / 2), int(y_range / 2))

    M = cv2.getRotationMatrix2D(center, -angel, 1.0)
    rotated = cv2.warpAffine(pic, M, (w, h))
    self.img = rotated
    cv2.imwrite(self.savepath, self.img)

    '''
    base = np.ones((3, 3))
    wrap = np.ones((3, 2))
    for i in range(3):
        base[i][1], base[i][2] = bas[i]
        wrap[i][0], wrap[i][1] = wra[i]
    A_abs = np.linalg.det(base)
    B = np.linalg.inv(base)
    base_ = B / A_abs
    coeff = np.dot(np.dot(np.dot(base.transpose(), base), base_), wrap)
    a1 = np.dot(base.transpose(), base)
    B = np.linalg.inv(a1)
    coeff = np.dot(np.dot(B, base.transpose()), wrap)

    a, b = self.baseimg.shape[0], self.baseimg.shape[1]
    test = np.zeros((4, 3))
    test[0][0] = 1
    test[1][0] = 1
    test[2][0] = 1
    test[3][0] = 1
    test[1][2] = b
    test[2][1] = a
    test[3][1] = a
    test[3][2] = b

    res = np.dot(test, coeff)
    #print(res)
    x_range = max(res[:, 0]) - min(res[:, 0])
    y_range = max(res[:, 1]) - min(res[:, 1])
    #print(math.ceil(x_range), math.ceil(y_range))

    newimg = np.zeros((self.wrapimg.shape[0], math.ceil(x_range), math.ceil(y_range)))
    '''
    return












def auto(self):#manual
    self.baseimg = gdalnumeric.LoadFile(self.base)
    self.wrapimg = gdalnumeric.LoadFile(self.wrap)
    dataSet = []
    bas = []
    wra = []
    with open(self.GCP) as fr:
        for line in fr.readlines()[5:8]:
            curline = line.strip()
            test = curline.split()
            x1, y1 = float(test[0]), float(test[1])
            x2, y2 = float(test[2]), float(test[3])
            bas.append((x1, y1))
            wra.append((x2, y2))
            dataSet.append(test)
    wrap = np.ones((3, 3))
    base = np.ones((3, 2))
    for i in range(3):
        wrap[i][1], wrap[i][2] = wra[i]
        base[i][0], base[i][1] = bas[i]

    a1 = np.dot(wrap.transpose(), wrap)
    B = np.linalg.inv(a1)
    coeff = np.dot(np.dot(B, wrap.transpose()), base)  # wrap:  (6, 498, 414) to  base:  (1390, 1071)
    # print(coeff)

    a, b = self.wrapimg.shape[1], self.wrapimg.shape[2]
    test = np.zeros((4, 3))
    test[0][0] = 1
    test[1][0] = 1
    test[2][0] = 1
    test[3][0] = 1
    test[1][2] = b
    test[2][1] = a
    test[3][1] = a
    test[3][2] = b
    res = np.dot(test, coeff)
    x_range = max(res[:, 0]) - min(res[:, 0])  # new image
    y_range = max(res[:, 1]) - min(res[:, 1])

    wrap = np.ones((3, 2))
    base = np.ones((3, 3))
    for i in range(3):
        wrap[i][0], wrap[i][1] = wra[i]
        base[i][1], base[i][2] = bas[i]

    ra = np.dot(base.transpose(), base)
    rb = np.linalg.inv(ra)
    rcoe = np.dot(np.dot(rb, base.transpose()), wrap)  # base to wrap

    delta_x = abs(min(res[:, 0]))  # delta of ordinate
    delta_y = abs(min(res[:, 1]))
    xy_list = [(1, i, j) for i in range(int(x_range)) for j in range(int(y_range))]
    xy = np.array(xy_list)
    xy[:, 1] = xy[:, 1] - delta_x
    xy[:, 2] = xy[:, 2] - delta_y

    wrap_pix_loc = np.dot(xy, rcoe)  # 1594*1409  *  2

    f = open("check.txt", "w")
    f.write(str(wrap_pix_loc[:1000]))
    f.close()

    newimg = np.zeros((int(x_range) * int(y_range), self.wrapimg.shape[0]))  # 6*1594*1409   2245946*6
    self.wrapimg = self.wrapimg.reshape(
        (self.wrapimg.shape[1] * self.wrapimg.shape[2], self.wrapimg.shape[0]))  # 206172*6
    for i in range(xy.shape[0]):
        if wrap_pix_loc[i][0] < 0 or wrap_pix_loc[i][1] < 0 or wrap_pix_loc[i][0] > self.wrapimg.shape[0] or \
                wrap_pix_loc[i][1] > self.wrapimg.shape[1]:
            pass
        else:
            # print(round(wrap_pix_loc[i][0]*wrap_pix_loc[i][1]))
            newimg[i, :] = self.wrapimg[int(round(wrap_pix_loc[i][0] * wrap_pix_loc[i][1])), :]

    R = newimg[:, 0].reshape((int(x_range), int(y_range)))
    G = newimg[:, 1].reshape((int(x_range), int(y_range)))
    B = newimg[:, 2].reshape((int(x_range), int(y_range)))
    self.img = cv2.merge([R, G, B])
    cv2.imwrite(self.savepath, self.img)
    print(1)

    p1 = compute.Point(res[0][0], res[0][1])
    p2 = compute.Point(res[1][0], res[1][1])
    p3 = compute.Point(res[2][0], res[2][1])
    p4 = compute.Point(res[3][0], res[3][1])
    fx = (compute.Getlen(p1, p3).getlen() + compute.Getlen(p2, p4).getlen()) / 2
    fy = (compute.Getlen(p1, p2).getlen() + compute.Getlen(p3, p4).getlen()) / 2

    self.img = np.transpose(self.wrapimg, [1, 2, 0]).astype(self.wrapimg.dtype)
    pic = cv2.resize(self.img[:, :, :3], (int(fx), int(fy)))

    # angel
    ang = (res[0][1] - test[0][2]) / (res[0][0] - test[0][1])
    angel = math.atan(ang) * 180 / (math.pi)
    (h, w) = (int(x_range), int(y_range))
    # center = (int(x_range / 2), int(y_range / 2))
    center = np.dot([1, int(self.wrapimg.shape[1] / 2), int(self.wrapimg.shape[2] / 2)], coeff)

    # rotate
    src = imgpro.Img(pic, h, w, [int(pic.shape[0] / 2), int(pic.shape[1] / 2)])
    src.Rotate(math.radians(angel))
    src.Process()
    cv2.imwrite("1.png", src.dst)
    cv2.waitKey(0)

    # M = cv2.getRotationMatrix2D(center, -angel, 1.0)
    # rotated = cv2.warpAffine(pic, M, (w, h))
    # self.img = rotated
    # cv2.imwrite(self.savepath, self.img)

    '''
    base = np.ones((3, 3))
    wrap = np.ones((3, 2))
    for i in range(3):
        base[i][1], base[i][2] = bas[i]
        wrap[i][0], wrap[i][1] = wra[i]
    A_abs = np.linalg.det(base)
    B = np.linalg.inv(base)
    base_ = B / A_abs
    coeff = np.dot(np.dot(np.dot(base.transpose(), base), base_), wrap)
    a1 = np.dot(base.transpose(), base)
    B = np.linalg.inv(a1)
    coeff = np.dot(np.dot(B, base.transpose()), wrap)

    a, b = self.baseimg.shape[0], self.baseimg.shape[1]
    test = np.zeros((4, 3))
    test[0][0] = 1
    test[1][0] = 1
    test[2][0] = 1
    test[3][0] = 1
    test[1][2] = b
    test[2][1] = a
    test[3][1] = a
    test[3][2] = b

    res = np.dot(test, coeff)
    #print(res)
    x_range = max(res[:, 0]) - min(res[:, 0])
    y_range = max(res[:, 1]) - min(res[:, 1])
    #print(math.ceil(x_range), math.ceil(y_range))

    newimg = np.zeros((self.wrapimg.shape[0], math.ceil(x_range), math.ceil(y_range)))
    '''
    return 
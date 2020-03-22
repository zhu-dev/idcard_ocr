# -*- coding: utf-8 -*-
import numpy as np
import cv2, time


class findidcard:
    def __init__(self):
        pass

    # img1为身份证模板, img2为需要识别的图像
    def find(self, img2_name):
        print(u'加载身份证照片中...')

        # 加载掩模图片和目标图片
        # 掩模图片用于与目标图匹配，矫正待识别图
        # 调整合适的图片比例
        img1_name = './mask/idcard_mask.jpg'
        img1 = cv2.UMat(cv2.imread(img1_name, 0))  # queryImage in Gray
        img1 = self.img_resize(img1, 640)
        # self.showimg(img1)
        # img1 = idocr.hist_equal(img1)

        img2 = cv2.UMat(cv2.imread(img2_name, 0))  # trainImage in Gray
        # print(img2.get().shape)
        img2 = self.img_resize(img2, 1920)
        # img2 = idocr.hist_equal(img2)
        # self.showimg(img2)

        img_org = cv2.UMat(cv2.imread(img2_name))
        # self.showimg(img_org)
        img_org = self.img_resize(img_org, 1920)

        #  Initiate SIFT detector
        t1 = round(time.time() * 1000)

        # 提取sift特征并匹配
        # 参考：https://blog.csdn.net/yan456jie/article/details/52312253
        # 获取SIFT对象
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        # 在图像中找到关键点，计算kp,des
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # 单应性匹配：两幅图像中的一幅 出现投影畸变的时候，他们还能彼此匹配
        # 单应性：指的是图像在投影发生了畸变后仍然能够有较高的检测和匹配准确率
        # 原理参考：https://www.cnblogs.com/Lin-Yi/p/9435824.html

        # kdtree建立索引
        FLANN_INDEX_KDTREE = 0  # kdtree建立索引方式的常量参数
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=10)  # checks指定索引树要被遍历的次数
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)  # 进行匹配搜索

        # store all the good matches as per Lowe's ratio test.
        # 两个最佳匹配之间距离需要大于ratio 0.7,距离过于相似可能是噪声点
        good = []
        # 遍历所有匹配组
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        # 通过一一对应的点，计算映射矩阵M，进行变换
        # 透视变换

        # reshape为(x,y)数组
        # 最小匹配数量设为10个
        MIN_MATCH_COUNT = 10
        if len(good) > MIN_MATCH_COUNT:
            # 获取关键点的坐标
            # src_pts原图中的标志点，source_points
            # dst_pts 目标图像中的标志点，target_points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # 用HomoGraphy计算图像与图像之间映射关系, M为转换矩阵
            # 原理详情参考：https://blog.csdn.net/xull88619814/article/details/81587595
            # 参数：https://blog.csdn.net/ei1990/article/details/78338928
            # 第三个参数：用来计算变换矩阵的方法
            # 第四个参数：原图像的点经过变换后点与目标图像上对应点的误差范围，超过就判定为outlier
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # 使用转换矩阵M计算出img1在img2的对应形状
            h, w = cv2.UMat.get(img1).shape
            M_r = np.linalg.inv(M) # 矩阵求逆，这里不理解，不取逆好像也可以
            # 透视变换，原理理解参考：https://blog.csdn.net/stf1065716904/article/details/92795238
            # 直观理解 参考：https://blog.csdn.net/dcrmg/article/details/80273818
            im_r = cv2.warpPerspective(img_org, M_r, (w, h))
            # self.showimg(im_r)
            # 到此，图片的预处理就完成了
        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        t2 = round(time.time() * 1000)
        # print(u'查找身份证耗时:%s' % (t2 - t1))
        return im_r

    def showimg(self, img):
        """调试时用来显示图片"""
        cv2.namedWindow("contours", 0)
        # cv2.resizeWindow("contours", 1600, 1200);
        cv2.imshow("contours", img)
        cv2.waitKey()

    def img_resize(self, imggray, dwidth):
        """按照比例调整图片的宽高"""
        # print 'dwidth:%s' % dwidth
        crop = imggray
        size = crop.get().shape
        height = size[0]
        width = size[1]
        height = height * dwidth / width
        crop = cv2.resize(src=crop, dsize=(dwidth, int(height)), interpolation=cv2.INTER_CUBIC)
        return crop

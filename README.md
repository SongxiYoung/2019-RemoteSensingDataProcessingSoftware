# 2019-RemoteSensingDataProcessingSoftware

This coursework used Qt and Python to design a remote sensing data processing software, including the user interface.

##FILE DESCRIPTION:

###RSPro.py:

Class RSImgPro is a subclass of QMainWindow. Object img is used to store the image matrix of the current window, with all processing operations using this member, ensuring that the image can be repeatedly processed.

![image](https://github.com/SongxiYoung/2019-RemoteSensingDataProcessingSoftware/blob/main/img-folder/1.png)


###equalHist.py:

Histogram equalization is essentially a nonlinear stretching of the image, redistributing the image pixel values so that the number of pixel values in a certain gray scale range is approximately equal.

![image](https://github.com/SongxiYoung/2019-RemoteSensingDataProcessingSoftware/blob/main/img-folder/%E5%9B%BE%E7%89%87%2018.png)


###convolve.py:

Suppose f(x,y) is the original image and g(x,y) is the filter. After aligning the original image with the window of the filter, the elements at their corresponding positions are multiplied and the results obtained are accumulated, and the final value obtained is the result obtained after filtering, whose position is located at the center of the window when the original image is aligned with the filter. This operation is the image filtering operation.

The purpose of filtering are: 1) to extract object features for image recognition, 2) to remove noise from the image.

![image](https://github.com/SongxiYoung/2019-RemoteSensingDataProcessingSoftware/blob/main/img-folder/%E5%9B%BE%E7%89%87%2019.png)


###PCA.py:

PCA transformation is a linear transformation applied to remote sensing images.

There is often a certain degree of correlation between the bands of multispectral data, and the application of PCA can remove the correlation, highlight the features, compress the data and eliminate the noise.

![image](https://github.com/SongxiYoung/2019-RemoteSensingDataProcessingSoftware/blob/main/img-folder/%E5%9B%BE%E7%89%87%2020.png)


###HSI.py:

The HSI transform fusion method consists of three main steps.
1, the original multispectral image is HSI transformed, 2, the I component of the multispectral image is directly replaced by the panchromatic image, keeping H and S unchanged, and 3, the enhanced multispectral image is obtained by HSI inverse transform.

![image](https://github.com/SongxiYoung/2019-RemoteSensingDataProcessingSoftware/blob/main/img-folder/hsi.png)


###NaiveBayesMulti.py:

![image](https://github.com/SongxiYoung/2019-RemoteSensingDataProcessingSoftware/blob/main/img-folder/BayesMulti_result.jpg)


###Classification_ISO.py:
![image](https://github.com/SongxiYoung/2019-RemoteSensingDataProcessingSoftware/blob/main/img-folder/ISODATA_test2.jpg)


###Classification_K.py:
![image](https://github.com/SongxiYoung/2019-RemoteSensingDataProcessingSoftware/blob/main/img-folder/Kmeans_result.jpg)


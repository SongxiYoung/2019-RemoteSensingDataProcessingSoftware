import numpy
from PIL import Image
import Classification_K

def load(filename):
    file = open(filename)
    data = []
    for eachline in file:
        linedata = eachline.strip().split(",")
        data.append( [ float(linedata[0]), float(linedata[1]), float(linedata[2]), float(linedata[3]), int(linedata[4]) ] )
    data = numpy.array(data)
    return data 

def mean_data(array):
    m, n = array.shape
    mean = numpy.zeros( (1, n-1) )
    for j in range(n-1):
        summ = 0
        for i in range(m):
            summ += array[i, j]
        mean[0][j] = summ / m
    return mean

def Std(data):
    m, n = data.shape
    m1 = mean_data(data)
    D = numpy.zeros( (1, n-1) )
    for j in range(n-1):
        sum_square = 0
        for i in range(m):
            sum_square += pow( (data[i][j] - m1[0][j]), 2 )
        D[0][j] = sum_square / float(m)
    return m1, numpy.sqrt(D)

def spiltData(train_data):
    m, n = train_data.shape
    classOne = []
    classTwo = []
    classThree = []
    for i in range(m):
        if train_data[i, -1] == 1:
            classOne.append( train_data[i] )
        elif train_data[i, -1] == 2:
            classTwo.append( train_data[i] )
        else:
            classThree.append( train_data[i] )
    return classOne, classTwo, classThree

def summarize(data_old):
    m = len(data_old)
    data = numpy.array(data_old).reshape( (m, 4) )
    ave, devi = Std(data)
    return ave, devi

def Gaussian_prob(x, ave, devi):
    return (1 / (numpy.sqrt(2*numpy.pi) * devi)) * numpy.exp(- numpy.power( (x-ave), 2 ) / (2*numpy.power(devi, 2) ))

def predict_class(eachdata, ave_one, devi_one, ave_two, devi_two, ave_three, devi_three):
    pred1_set = Gaussian_prob(eachdata, ave_one, devi_one)
    m, n = pred1_set.shape
    pred1 = 1
    for i in range(n):
        pred1 = pred1 * pred1_set[0][i]

    pred2_set = Gaussian_prob(eachdata, ave_two, devi_two)
    pred2 = 1
    for i in range(n):
        pred2 = pred2 * pred2_set[0][i]

    pred3_set = Gaussian_prob(eachdata, ave_three, devi_three)
    pred3 = 1
    for i in range(n):
        pred3 = pred3 * pred3_set[0][i]

    if pred1 > pred2:
        if pred1 > pred3:
 #           print("Probability for class 1: ",  pred1 )
            return 1
        else:
 #           print("Probability for class 3: ",  pred3 )
            return 3

    else:
        if pred3 > pred2:
 #           print("Probability for class 3: ",  pred3 )
            return 3
        else:
 #           print("Probability for class 2: ",  pred2 )
            return 2

def predict(test_data, ave_one, devi_one, ave_two, devi_two, ave_three, devi_three):
    m = len(test_data)
    data_old = numpy.array(test_data).reshape( (m, 3) )
    data = numpy.zeros( (m, 3) )
    prediction = numpy.zeros( (m, 1) )
    for j in range(3):
        data[:,j] = data_old[:,j]
    for i in range(m):
        prediction[i] = predict_class(data[i], ave_one, devi_one, ave_two, devi_two, ave_three, devi_three)
    return prediction

def accuracy(labels, prediction):
    m, n = prediction.shape
    summ = 0
    for i in range(m):
        for j in range(n):
            if labels[i][j] == prediction[i][j]:
                summ += 1
    return summ/m

if __name__ == '__main__':
    read = Image.open('M.jpg')
    lena = numpy.array(read)

    m = lena.shape[0]*lena.shape[1]
    n = lena.shape[2]
    data = lena.reshape(m, n)
    centers = Classification_K.randinitialize(data, 3)      #预估类别数
    new, classes, result = Classification_K.Kmeans(data, centers, 3)

    train_data = numpy.empty((m, n+1))
    for i in range(m):
        for j in range(n):
            train_data[i, j] = data[i, j]
            train_data[i, j+1] = result[i]

    test_data = data.copy()

    classOne, classTwo, classThree = spiltData(train_data)
    ave_one, devi_one = summarize(classOne)
    ave_two, devi_two = summarize(classTwo)
    ave_three, devi_three = summarize(classThree)
    prediction = predict(test_data, ave_one, devi_one, ave_two, devi_two, ave_three, devi_three)

    k = 0
    for i in range(lena.shape[0]):
        for j in range(lena.shape[1]):
            if result[k]==0:
                lena[i, j, :] = [255, 0, 0]
            if result[k]==1:
                lena[i, j, :] = [255, 250, 250]
            if result[k]==2:
                lena[i, j, :] = [255, 255, 0]
            if result[k]==3:
                lena[i, j, :] = [255, 192, 203]
            if result[k]==4:
                lena[i, j, :] = [160, 32, 240]
            if result[k]==5:
                lena[i, j, :] = [255, 165, 0]
            else:
                pass
            k = k + 1
            
    new_img = Image.fromarray(lena) 
    new_img.show()
    new_img.save("BayesMulti_result.jpg")
'''
data = load('iris.txt')
random.shuffle(data)

train_data = data[:120]                       #100*5
test_data = data[120:]                        #50*5

labels = numpy.zeros( (30, 1) )
for i in range(30):
    labels[i] = test_data[i, -1]


classOne, classTwo, classThree = spiltData(train_data)
ave_one, devi_one = summarize(classOne)
ave_two, devi_two = summarize(classTwo)
ave_three, devi_three = summarize(classThree)
prediction = predict(test_data, ave_one, devi_one, ave_two, devi_two, ave_three, devi_three)
accu = accuracy(labels, prediction)
print("Accuracy:", accu*100, "%")
'''
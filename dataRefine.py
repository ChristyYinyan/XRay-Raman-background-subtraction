__author__ = 'yinyan'
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filepath="HighPHEA_Long300_CombiView"
data=pd.read_csv("OriginalTxtDataSet/HighPHEA_Long300_CombiView.csv", header=None)
data=data.as_matrix()
sampleNumber=107
data=data[:sampleNumber+1]


def std():
    """
    read the xsl file and return the variance plot and peak positions
    :return:
    """
    theta=data[0] #title
    print "theta's length", len(theta)
    # print "Each sample ha0301HighPHEA CombiView File RTs ", len(theeta), "data points"
    # print "There are ", len(data[1:,0]), " samples in total"

    """1:Plot all samples"""
    for i in range(1, sampleNumber+1):
        plt.plot(theta, data[i])
    plt.savefig(filepath+"/original all samples")
    plt.clf()

    # plt.plot(theta, data[3])
    # plt.savefig(filepath+"/sample2")
    # plt.clf()

    """2:Plot std of each column"""
    stdValue=[]
    for i in range(len(data[0])):
        stdValue.append(np.std(data[1:, i]))#get the std of each column
    """2:plt original std values"""
    plt.plot(stdValue)
    plt.savefig(filepath+"/original std values's plot")
    plt.clf()

    """3:std value substraction"""
    for i in range(len(stdValue)-1):
        stdValue[i]=abs(stdValue[i]-stdValue[i+1])
    stdValue[-1]=abs(stdValue[-1]-stdValue[-2])

    """3:plot std values after subtraction"""
    plt.plot(stdValue)
    plt.savefig(filepath+"/std values 's plot after subtraction")
    plt.clf()


    """4: filter out the highest std peaks position"""
    sortStd=sorted(stdValue, reverse=True)
    bValue=len(theta)/10
    highest=sortStd[:bValue] #save the highest 50 std values
    print "The lowest peak which is picked", highest[-1]
    # print "The highest " + str(bValue) + " std values", highest
    """4: plot the highest 'bValue' peaks"""
    plt.plot(highest)
    plt.savefig(filepath + "/highest " + str(bValue)+ " Std Value Plot.png")
    plt.clf()

    highestTheta=[]
    for i in range(len(stdValue)):
        if stdValue[i] in highest:
            highestTheta.append(theta[i])
    # print "The highest " + str(bValue) + " of", highestTheta


    """Highest STD value positions"""
    position=[]
    for i in range(len(stdValue)):
        if stdValue[i] in highest:
            position.append(i)
    print "The highest std values' positions are ", position
    return position, highestTheta

#Average the data point
def refine_data(data, position):
    new_data=data[1:, position] #refined data
    meanValue=np.copy(new_data[:,0])
    count=1.0
    for i in range(1,new_data.shape[1]):
        if position[i]-position[i-1]>2:
            meanValue=meanValue/count
            for j in range(int(count)):
                new_data[:,i-j-1]=new_data[:,i-j-1]-meanValue
            meanValue=np.copy(new_data[:,i])
            count=1.0
        else:
            count+=1.0
            meanValue+=new_data[:,i]
        if i==new_data.shape[1]-1:
            meanValue=meanValue/count
            for j in range(int(count)):
                new_data[:,i-j]=new_data[:,i-j]-meanValue
    return new_data

#resolve split of position
def split_position(position):
    intersectS=[]#every block's start point
    intersectE=[]#every block's end point
    single=[]
    intersectS.append(position[0])
    distance=3
    for i in range(1, len(position), 1):
        if position[i]-position[i-1]>=distance:
            intersectE.append(position[i-1])
            intersectS.append(position[i])
            if i<len(position)-1:
                if position[i+1]-position[i]>distance and position[i]-position[i-1]>distance:
                    single.append(position[i])
    intersectS.pop()
    print len(intersectE)
    print "start", intersectS
    print "end", intersectE
    print "single", single
    return intersectS, intersectE, len(position)-len(single)-1

def polyData(data, s, e, position):
    rangeLength=len(s)
    neighborNum=4
    theta=data[0]
    second, m=[[]]*rangeLength, 0 #use to save left 4 neighbors and right 4 neighbors
    secondX=[[]]*rangeLength
    secondOrigin=[[]]*rangeLength
    secondOriginX=[[]]*rangeLength


    for i in range(len(s)):
        temp1=range(s[i]-neighborNum, s[i])
        temp2=range(e[i]+1, e[i]+neighborNum+1)
        temp=temp1+temp2
        second[m]=data[1:, temp]
        secondX[m]=theta[temp]
        tp=[]
        for num in position:
            if num in range(s[i], e[i]+1):
                tp.append(num)
        secondOrigin[m]=data[1:, tp]
        secondOriginX[m]=theta[tp]
        m+=1
    return second, secondOrigin, secondX, secondOriginX

#2nd order, first order, single polynomial linear regression
def refine_secondOrder(second, secondOrigin, secondX, secondOriginX, l):
    for i in range(len(second)):
        for j in range(len(second[i])):
            z=np.polyfit(secondX[i], second[i][j], 2)
            p=np.poly1d(z)
            tempX=secondOriginX[i]
            tempY=secondOrigin[i][j]
            for k in range(len(tempX)):
                tempY[k]-=p(tempX[k])
            secondOrigin[i][j]=tempY


    """find out the odd sample, which is sample 2"""
    delete=[]
    highestPeak=200
    lowestPeak=-10
    for j in range(sampleNumber):
        for i in range(len(secondOrigin)):
            for k in range(len(secondOrigin[i][j])):
                if secondOrigin[i][j][k]>highestPeak or secondOrigin[i][j][k]<lowestPeak:
                   delete.append(j)
    delete=list(set(delete))
    print "the deleted samples are", delete

    """plot the refine data"""
    result=np.zeros((l+1,sampleNumber))
    # print secondOriginX
    print "length of secondorigin x cordinate", len(secondOriginX)
    a=secondOriginX[0]
    for i in range(1, len(secondOriginX)):
        a=np.concatenate((a, secondOriginX[i]), axis=None)

    for i in range(sampleNumber):
        index=0
        for j in range(len(secondOrigin)):
            length=secondOrigin[j][i].shape[0]
            # print length
            result[index:index+length,i]=secondOrigin[j][i]
            # print index
            index=index+length

    # print result.shape
    result=np.delete(result, (delete), axis=1)
    return result, delete
def new_data_write(result, data,  position, delete):
    position.pop()
    length=len(data[0])
    result=result.transpose()
    # print result.shape
    m=0
    for i in range(1, sampleNumber+1):
        if i in delete:
            data[i]=np.zeros(length)
        else:
            k=0
            for j in range(length):
                if j in position:
                    data[i][j]=result[m][k]
                    k+=1
                else:
                    data[i][j]=0
            m+=1

    writer=csv.writer(open(filepath+"/refine_HighPHEA_CombiView_File_RT.csv","w"), delimiter=',',quoting=csv.QUOTE_ALL)
    writer.writerow([float(x) for x in data[0]])
    for i in range(1, sampleNumber+1, 1):
        writer.writerow([float(y) for y in data[i]])
    for i in range(1, sampleNumber+1, 1):
        plt.plot(data[0], data[i])
    plt.savefig(filepath + "/plot result after 2nd order approximation")
    plt.clf()


if __name__=="__main__":
    position, theta=std()
    new_data=refine_data(data, position)
    s, e, l=split_position(position)
    second, secondOrigin, secondX, secondOriginX=polyData(data, s, e, position)
    result, delete=refine_secondOrder(second, secondOrigin, secondX, secondOriginX, l)
    new_data_write(result, data,position, delete)





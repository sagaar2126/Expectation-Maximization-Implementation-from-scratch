import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from numpy.linalg import det,inv
import math
train_a = pd.DataFrame(pd.read_csv('P3a_train_data.txt',sep=",",header=None, index_col=False ))
test_a = pd.DataFrame(pd.read_csv('P3a_test_data.txt',sep=",",header=None, index_col=False ))

def Mean(sub_data,N,data,d):
    indexes=list(sub_data.sample(n=N).index)
    # print(indexes)
    mu=[]
    # print(len(indexes))
    for i in range(d):
        sum_=0
        for j in indexes:
            sum_=sum_+data.iloc[j][i]
        mu.append(sum_/N) 
    print(mu)    
    return np.array(mu).reshape(-1,1) , indexes    
  
def covariances(data, indexes , mu,d):
    C=np.zeros((d,d))
    for i in range(len(indexes)):
        A=np.array(data.iloc[indexes[i]][:d]).reshape(-1,1)
        C=C+(A-mu)*(A-mu).T    
    return C/len(indexes)    
    

def fit(mu1,mu2,C1,C2,data,d):
    correct_1=[]       # correctly classified as -1
    incorrect_1=[]     # incorrectly classified as -1
    correct_2=[]       # correctly classified as 1
    incorrect_2=[]     # incorrectly classified as 1
    for i in range(len(data)):
        x=np.array(data.iloc[i][:d]).reshape(-1,1)
        a=np.matmul(np.matmul((x-mu1).T,inv(C1)),(x-mu1))
        b=np.matmul(np.matmul((x-mu2).T,inv(C2)),(x-mu2))
        dist_1=(1/math.sqrt(det(C1)))*math.exp(-0.5*a)
        dist_2=(1/math.sqrt(det(C2)))*math.exp(-0.5*b)
        if(dist_1>dist_2):
            if(data.iloc[i][d]==-1):
                correct_1.append(i)
            else:
                incorrect_1.append(i)
        else:
            if(data.iloc[i][d]==1):
                correct_2.append(i)
            else:
                incorrect_2.append(i)
        
    return correct_1, incorrect_1 ,correct_2 ,incorrect_2  
 
def bayes():
    d=len(train_a.columns)-1
    class_1=train_a[train_a[d]==-1]
    class_2=train_a[train_a[d]==1]
    per=[]
    index=[]
    
    mu1,index1=Mean(class_1,2,train_a,d)
    mu2,index2=Mean(class_2,3,train_a,d)
    C1=covariances(train_a,index1,mu1,d)
    C2=covariances(train_a,index2,mu2,d)
    A,B,C,D = fit(mu1,mu2,C1,C2,test_a,d)
    print(C1)
    print(C2)
    accuracy= ((len(A)+len(C))/len(test_a))*100
    
   
    print(accuracy)
    # plt.plot(index,per)
    # plt.gca()
    # plt.show()    
    
    
def fit_knn(train,k,test,d,indexes):
    
    correct=[]
    
    for i in range(len(test)):
        
        distance=[]
        x=np.array(test.iloc[i][:d]).reshape(-1,1)
        for j in indexes:
            a=np.array(train.iloc[j][:d]).reshape(-1,1)
            dist=np.sqrt(np.sum(np.square(a-x)))
            distance.append(dist)
        
        arr=np.array(distance)
        sort_index=np.argsort(arr)
        count_1=0
        count_2=0
        
        
        for l in sort_index[:k]:
            if train.iloc[indexes[l]][d]==-1:
                count_1+=1
            if train.iloc[indexes[l]][d]==1:
                count_2+=1
        
        if(count_1>count_2):
            
            if test.iloc[i][d]==-1:
                correct.append(i)
               
        else:
            if test.iloc[i][d]==1:
                correct.append(i)
    return correct
    
    
def knn():
    d=len(train_a.columns)-1
    N=75
    class_1=train_a[train_a[d]==-1]
    class_2=train_a[train_a[d]==1]
    index1 =list(class_1.sample(n=N//2).index)
    index2 =list(class_2.sample(n=N//2).index)
    indexes=index1+index2
    correct=fit_knn(train_a,1,test_a,d,indexes)
    accuracy=(len(correct)/len(test_a))*100
    
    print(accuracy)
    
def EM():
    
    d=len(train_a.columns)-1
    data=train_a.iloc[:,d]
    mu1=np.random.normal(0,1,(d,1))
    mu2=np.random.normal(0,1,(d,1))
    print(mu1.shape)
    print(mu2.shape)
    # C1=np.random.normal(0,1,(d,d))
    # C2=np.random.normal(0,1,(d,d))
    
    C1=np.array([[1,0],[0,1]])
    C2=np.array([[1,0],[0,1]])
    Lambda1=0.5
    Lambda2=0.5
    for i in range(10):
        gama1=[]
        gama2=[]
        mu1_new=0
        mu2_new=0
        for j in range(len(train_a)):
            x=np.array(train_a.iloc[j][:d]).reshape(-1,1)
            print(x.shape)
            a=np.matmul(np.matmul((x-mu1).T,inv(C1)),(x-mu1))
            b=np.matmul(np.matmul((x-mu2).T,inv(C2)),(x-mu2))
            phi1=(1/math.sqrt(abs(det(C1))))*math.exp(-0.5*a)
            phi2=(1/math.sqrt(abs(det(C2))))*math.exp(-0.5*b)
            g1=(Lambda1)*(phi1)/(Lambda1*phi1+Lambda2*phi2)
            g2=(Lambda2)*(phi2)/(Lambda1*phi1+Lambda2*phi2)
            gama1.append(g1)
            gama2.append(g2)
            mu1_new=g1*x+mu1_new
            mu2_new=g2*x+mu2_new
        mu1=mu1_new/sum(gama1)
        mu2=mu2_new/(sum(gama2))
        C1_new=0
        C2_new=0
        for j in range(len(train_a)):
            x=np.array(train_a.iloc[j][:d]).reshape(-1,1)
            C1_new=C1_new+gama1[j]*np.matmul((x-mu2),(x-mu1).T)
            C2_new=C2_new+gama1[j]*np.matmul((x-mu2),(x-mu1).T)
        C1=C1_new/sum(gama1)
        C2=C2_new/(sum(gama2))
        Lambda1=sum(gama1)/(len(train_a))
        Lambda2=sum(gama2)/(len(train_a))
    A,B,C,D = fit(mu2,mu1,C2,C1,test_a,d)
    accuracy= ((len(A)+len(C))/len(test_a))*100
    print(mu1)
    print(mu2)
    print(C1)
    print(C2)
    print(accuracy)
    
def Exponential():
    d=len(train_a.columns)-1
    class_1=train_a[train_a[d]==-1]
    class_2=train_a[train_a[d]==1]
    mu1,index1=Mean(class_1,100,train_a,d)
    C1=covariances(train_a,index1,mu1,d)
    mu2,index2=Mean(class_2,100,train_a,d)
    lambda1,lambda2=(1/mu2[0,0]),(1/mu2[1,0])
    while (not(lambda1>0 and lambda2>0)):
        mu2,index2=Mean(class_2,100,train_a,d)
        lambda1,lambda2=(1/mu2[0,0]),(1/mu2[1,0])
    correct_1=[]
    incorrect_1=[]
    correct_2=[]
    incorrect_2=[]   
    data=test_a
    print(lambda1)
    print(lambda2)
    for i in range(len(data)):
        x=np.array(data.iloc[i][:d]).reshape(-1,1)
        a=np.matmul(np.matmul((x-mu1).T,inv(C1)),(x-mu1))
        dist_1=(1/math.sqrt(det(C1)))*math.exp(-0.5*a)
        dist_2=(lambda1*lambda2)*math.exp(-(lambda1*data.iloc[i][0])-(lambda2*data.iloc[i][1]))
        if(dist_1>dist_2):
            if(data.iloc[i][d]==-1):
                correct_1.append(i)
            else:
                incorrect_1.append(i)
        else:
            if(data.iloc[i][d]==1):
                correct_2.append(i)
            else:
                incorrect_2.append(i)
    
    accuracy= ((len(correct_1)+len(correct_2))/len(data))*100
    
    print(accuracy)
                
    
def main():
    # bayes()
    # knn()
    EM()
    # Exponential()
                
if __name__ == "__main__":
    main()
            


















    

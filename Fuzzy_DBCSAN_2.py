import  numpy  as  numpy
import  scipy  as  scipy
from  sklearn  import  cluster
import  matplotlib.pyplot  as  plt
  
     
def  set2List(NumpyArray):
    list  =  []
    for  item  in  NumpyArray:
        list.append(item.tolist())
    return  list   
  
def  member(value,MinumumPoints,Max_Points):
    if  value<=MinumumPoints:
        return  0
    elif  value<Max_Points:
        return  (value  -  MinumumPoints)/(Max_Points  -  MinumumPoints)
    else:
        return  1    
def Max_Distance(i,PointNeighbors,DistanceMatrix):
    maximum = 0
    for k in PointNeighbors:
        if DistanceMatrix[i][k] > maximum:
            maximum = DistanceMatrix[i][k]
    return maximum
            
def calculate_Membership(maximum,distance):
    return ((maximum - distance)/(maximum+distance))

def  DBSCAN(Dataset,Epsilon, Points,DistanceMethod  =  'euclidean'):
#    Dataset  is  a  mxn  matrix    m  is  number  of  item  and  n  is  the  dimension  of  data
    m,n = Dataset.shape
    # Visited = numpy.zeros(m,'int')
    Type = numpy.zeros(m)
    Membership = []
    for i in range(m):
        Membership.append({})
#      -1  noise    outlier
#    0  border
#    1  core
    ClustersList=[]
    Cluster=[]
    PointClusterNumber=numpy.zeros(m)
    PointClusterNumberIndex=1
    PointNeighbors=[]
    DistanceMatrix  =  scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Dataset,DistanceMethod))
    
    BorderPoint = []
    for  i  in  range(m):
        BorderPoint = []
        if  len(Membership[i]) == 0:
            PointNeighbors = numpy.where(DistanceMatrix[i]<=Epsilon)[0]
            PointNeighbors = set2List(PointNeighbors)

            Maximum = Max_Distance(i,PointNeighbors,DistanceMatrix)

            for k in PointNeighbors:
                if DistanceMatrix[i][k] == Maximum:        
                    BorderPoint.append(k)

            if len(PointNeighbors) >= Points:
                Membership[i][PointClusterNumberIndex] = 1
                Type[i] = 1
                ExpandCluster(DistanceMatrix,i,Epsilon,Points,PointNeighbors,BorderPoint,Membership,PointClusterNumberIndex,Type,Maximum)    
                PointClusterNumberIndex += 1
            else:
                Type[i] = -1
    print(PointClusterNumberIndex)
    return Membership,Type,PointClusterNumberIndex-1

def ExpandCluster(DistanceMatrix,PointtoExpand,Epsilon,Points,PointNeighbors,BorderPoint,Membership,PointClusterNumberIndex,Type,Maximum):
    Neighbors = []
    for i in PointNeighbors:
        if i not in  BorderPoint:
            Membership[i][PointClusterNumberIndex] = calculate_Membership(Maximum,DistanceMatrix[i][PointtoExpand])
    for i in BorderPoint:
        for j in Neighbors:
            if (j not in BorderPoint) and (PointClusterNumberIndex not in Membership[j]):
                Membership[j][PointClusterNumberIndex] = calculate_Membership(Maximum,DistanceMatrix[j][PointtoExpand])
            elif j not in BorderPoint:
                Membership[j][PointClusterNumberIndex] = max(Membership[j][PointClusterNumberIndex],calculate_Membership(Maximum,DistanceMatrix[j][PointtoExpand]))

        Neighbors = numpy.where(DistanceMatrix[i]<=Epsilon)[0] 
        Neighbors = set2List(Neighbors)

        Maximum = Max_Distance(i,Neighbors,DistanceMatrix)

        for k in Neighbors:
            if DistanceMatrix[i][k] == Maximum and k != PointtoExpand:
                try:
                    BorderPoint.index(k)
                except ValueError:    
                    BorderPoint.append(k) 
        if len(Neighbors) >= Points:
            Membership[i][PointClusterNumberIndex] = 1
            Type[i] = 1
        else:
            Membership[i][PointClusterNumberIndex] = 0
            Type[i] = 0        
        PointtoExpand = i
    return


def  calculate_PC(Data_len,cluster_size,Membership_value):
    pc  =  0
    for  i  in  range(Data_len):
        for j in range(cluster_size):
            if (j+1) in Membership_value[i]:
                pc  +=  Membership_value[i][j+1]**2
    print(pc)            
    return  pc/Data_len  
def  calculate_FPI(Data_len,cluster_size,Membership_value):
    value  =  0
    for  i  in  range(Data_len):
        for j in range(cluster_size):
            if (j+1) in Membership_value[i]:
                value  +=  (Membership_value[i][j+1]**2)/Data_len
    FPI  =  1  -  (cluster_size/(cluster_size-1))*(1-value)
    return  FPI          

def  F_Precision(Data,k,l,Output,Membership_value):
    value  =  0
    count  =  0
    for  i  in  range(len(Data)):
        if  (k+1) in Membership_value[i] :
            count+=1
            if   (k+1) == Output[i] :
                value  +=  Membership_value[i][k+1]
    if  count  !=  0:
        return  value/count,count
    else:
        return  0,1    
def  F_recall(Data,k,l,Output,Membership_value):
    value  =  0
    count  =  0
    for  i  in  range(len(Data)):
        if  Output[i]  ==  (k+1) :
            count+=1
            if    (k+1) in Membership_value[i]:
                value  +=  Membership_value[i][k+1]
    if  count  !=  0:
        return  value/count,count
    else:
        return  0,1  

X  =  numpy.loadtxt("glass.txt",dtype  =  'float')
Data  =  X[:,  1:-1]
Output  =  X[: ,-1]               
# Data  =  numpy.array([[1,2],[1.5,2.5],[0,1],[1,1],[1.8,0.5],[0.5,1.2],[4,4.2],[4,5.1],[4.8,3.9],[5.2,6.2],[5,5.5],[4.6,6.1],[3,3.2],[2.2,3],[2,2],[4.5,5],[3.8,3.8],[1.5,1.5],[5,4.5],[0,7],[2,2.5],[0.5,2],[2.5,2],[3.8,4.3],[4.5,4.5],[4,4],[1.5,2]])

Epsilon=2
Points=7

Membership_value,Type,cluster_len=DBSCAN(Data,Epsilon,Points)
print(Membership_value,"\n",Type)           
  
PC = calculate_PC(len(Data),cluster_len,Membership_value)
FPI = calculate_FPI(len(Data),cluster_len,Membership_value)
F_measure  =  0
l  =  list(set(Output))
print(l)
for  i  in  range(len(l)):
    f_Precision,c1  =  F_Precision(Data,i,l,Output,Membership_value)
    f_recall,c2  =  F_recall(Data,i,l,Output,Membership_value)
    # count_cluster_point  =  Count_cluster(i,result)
    # count_class_point  =  Count_Class(i,Output)
    if  c2 !=  0  and  (f_recall+f_Precision)  !=0:
        F_measure  +=  (c1/c2)*((f_Precision*f_recall)/(f_recall+f_Precision))

#printed  numbers  are  cluster  numbers
# ,round(PC,2),round(FPI,2), F_measure
print  (round(PC,2),round(FPI,2),F_measure)
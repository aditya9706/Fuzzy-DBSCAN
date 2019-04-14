import  numpy  as  numpy
import  scipy  as  scipy
from  sklearn  import  cluster
import  matplotlib.pyplot  as  plt
  
  
     
def  set2List(NumpyArray):
    list  =  []
    for  item  in  NumpyArray:
        list.append(item.tolist())
    return  list
  
  
# def  GenerateData():
#     x1=numpy.random.randn(50  2)
#     x2x=numpy.random.randn(80  1)+12
#     x2y=numpy.random.randn(80  1)
#     x2=numpy.column_stack((x2x  x2y))
#     x3=numpy.random.randn(100  2)+8
#     x4=numpy.random.randn(120  2)+15
#     z=numpy.concatenate((x1  x2  x3  x4))
#     return  z    
  
def  member(value,MinumumPoints,Max_Points):
    if  value<=MinumumPoints:
        return  0
    elif  value<Max_Points:
        return  (value  -  MinumumPoints)/(Max_Points  -  MinumumPoints)
    else:
        return  1    



  
def  DBSCAN(Dataset,Epsilon_min,Epsilon_max, MinumumPoints,Max_Points,DistanceMethod  =  'euclidean'):
#    Dataset  is  a  mxn  matrix    m  is  number  of  item  and  n  is  the  dimension  of  data
    m,n = Dataset.shape
    Visited = numpy.zeros(m,'int')
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
    # for i in DistanceMatrix:
    #     print(i)

    for  i  in  range(m):
        if  Visited[i] == 0:
            Visited[i] = 1
            PointNeighbors = numpy.where(DistanceMatrix[i]<Epsilon_max)[0]
            PointNeighbors = set2List(PointNeighbors)
            # print("DBSCAN  ::", i, "\n",PointNeighbors)
            

            density = 0
            for  k  in  PointNeighbors:
                membership  =  0
                if DistanceMatrix[i][k] <= Epsilon_min:
                    membership  =  1

                elif DistanceMatrix[i][k] <= Epsilon_max:
                    membership = ((Epsilon_max - DistanceMatrix[i][k])/(Epsilon_max - Epsilon_min))
                
                else:
                    membership  =  0

                density = density + membership      
            # print(density)
            if  member(density,MinumumPoints,Max_Points)  ==  0  :
                Type[i] = -1
                Visited[i] = 0

            else:
                for k in PointNeighbors:
                    if PointClusterNumber[k]!=PointClusterNumberIndex and Type[k] == 0 and k!=i:
                        Visited[k] = 0
                Cluster = []
                Cluster.append(i)
                PointClusterNumber[i] = PointClusterNumberIndex
                Type[i] = 1
                Membership[i][PointClusterNumberIndex] = round(member(density ,MinumumPoints ,Max_Points),2)
                
                ExpandClsuter(i,PointNeighbors,Cluster,MinumumPoints,Max_Points,Epsilon_min,Epsilon_max,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex,Type,Membership  )
                Cluster.append(PointNeighbors[:])
                ClustersList.append(Cluster[:])
                PointClusterNumberIndex=PointClusterNumberIndex+1
                  
    # print(Type,"\n",Membership)
    return  PointClusterNumber,PointClusterNumberIndex-1,Membership,Type  
  
  
  
def  ExpandClsuter(PointToExapnd,PointNeighbors,Cluster,MinumumPoints,Max_Points,Epsilon_min,Epsilon_max,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex,Type,Membership):
    Neighbors=[]

    for  i  in  PointNeighbors:
        if  Visited[i]==0:
            Visited[i]=1
            Neighbors=numpy.where(DistanceMatrix[i]<Epsilon_max)[0]
            # print("Expand::",i)
            # print(Neighbors)
            
            density  =  0
            for  k  in  PointNeighbors:
                membership  =  0
                if  DistanceMatrix[i][k]<=Epsilon_min:
                    membership  =  1

                elif  DistanceMatrix[i][k]<=Epsilon_max:
                    membership  =  ((Epsilon_max-DistanceMatrix[i][k])/(Epsilon_max-Epsilon_min))
                
                else:
                    membership  =  0
                density  =  density  +  membership
            m  =  round(member(density,MinumumPoints,Max_Points),2)
            # print(density)      
            if  m  >  0:
#                Neighbors  merge  with  PointNeighbors
                for k in Neighbors:
                    if PointClusterNumber[k]!=PointClusterNumberIndex and Type[k] == 0 and k!=i:
                        Visited[k] = 0
                for  j  in  Neighbors:
                    try:
                        PointNeighbors.index(j)
                    except  ValueError:
                        PointNeighbors.append(j)
                Type[i]  =  1
                Membership[i][PointClusterNumberIndex]  =  m
            else:
                Type[i]  =  0
                minimum  =  9999
                for  k  in  Neighbors:
                    if  k  !=  i:
                        if  DistanceMatrix[i][k]  <  Epsilon_min:
                            h  =  1
                        else  :
                            h  =  (Epsilon_max  -  DistanceMatrix[i][k])  /  (Epsilon_max  -  Epsilon_min)
                        Neighbors2=numpy.where(DistanceMatrix[k]<Epsilon_max)[0]
                        density  =  0
                        for  p  in  Neighbors2:
                            membership  =  0
                            if  DistanceMatrix[k][p]<=Epsilon_min:
                                membership  =  1

                            elif  DistanceMatrix[p][k]<=Epsilon_max:
                                membership  =  ((Epsilon_max-DistanceMatrix[i][k])/(Epsilon_max-Epsilon_min))
                            
                            else:
                                membership  =  0
                            density  =  density  +  membership
                        m  =  round(member(density,MinumumPoints,Max_Points),2)
                        mini  =  9999
                        if  m>0  and  h>0:
                            mini  =  min(m, round(h,2))
                        if  mini  <  minimum:
                            minimum  =  mini
                if minimum == 9999:
                    Membership[i][PointClusterNumberIndex]  =  0
                else:
                    Membership[i][PointClusterNumberIndex] = minimum                                    


        if  PointClusterNumber[i]==0:
            Cluster.append(i)
            PointClusterNumber[i]=PointClusterNumberIndex
    return
  

def  calculate_PC(Data_len,cluster_size,Membership_value):
    pc  =  0
    for  i  in  range(Data_len):
        for j in range(cluster_size):
            if (j+1) in Membership_value[i]:
                pc  +=  Membership_value[i][j+1]**2
    return  pc/Data_len  
def  calculate_FPI(Data_len,cluster_size,Membership_value):
    value  =  0
    for  i  in  range(Data_len):
        for j in range(cluster_size):
            if (j+1) in Membership_value[i]:
                value  +=  (Membership_value[i][j+1]**2)/Data_len
    FPI  =  1  -  (cluster_size/(cluster_size-1))*(1-value)
    return  FPI          

def  F_Precision(Data,k,result,l,Output,Membership_value):
    value  =  0
    count  =  0
    for  i  in  range(len(result)):
        if  k in Membership_value[i] :
            count+=1
            if   k == Output[i] :
                value  +=  Membership_value[i][k]
    if  count  !=  0:
        return  value/count,count
    else:
        return  0,1    
def  F_recall(Data,k,result,l,Output,Membership_value):
    value  =  0
    count  =  0
    for  i  in  range(len(result)):
        if  Output[i]  ==  k :
            count+=1
            if    k in Membership_value[i]:
                value  +=  Membership_value[i][k]
    if  count  !=  0:
        return  value/count,count
    else:
        return  0,1    
def  Count_cluster(k,result):
    count  =  0
    for  i  in  range(len(result)):
        if  result[i]  ==  k:
            count+=1
    return  count              
def  Count_Class(k,Output):
    count  =  0
    for  i  in  range(len(result)):
        if  Output[i]  ==  k:
            count+=1
    return  count    

def Display(Data_len,Membership_value,Type,cluster_len):
    Output = []
    for i in range(Data_len):
        Output.append({})
    for i in range(Data_len):
        if len(Membership_value[i]) == 0:
            Output[i][0] = []
            Output[i][0].append(-1)
        else:
            for j in range(cluster_len):
                if (j+1) in Membership_value[i]:
                    Output[i][j+1] = []
                    Output[i][j+1].append(Type[i])
                    Output[i][j+1].append(Membership_value[i][j+1])
    return Output                


#Generating  some  data  with  normal  distribution  at  
#(0  0)
#(8  8)
#(12  0)
#(15  15)
#  Data=GenerateData()

X  =  numpy.loadtxt("glass.txt",dtype  =  'float')
Data  =  X[:,  1:-1]
Output  =  X[: ,-1]
# Data  =  numpy.array([[1,2],[1.5,2.5],[0,1],[1,1],[1.8,0.5],[0.5,1.2],[4,4.2],[4,5.1],[4.8,3.9],[5.2,6.2],[5,5.5],[4.6,6.1],[3,3.2],[2.2,3],[2,2],[4.5,5],[3.8,3.8],[1.5,1.5],[5,4.5],[0,7],[2,2.5],[0.5,2],[2.5,2],[3.8,4.3],[4.5,4.5],[4,4],[1.5,2]])
                 
  
#Adding  some  noise  with  uniform  distribution  
#X  between  [-3  17]  
#Y  between  [-3  17]
#  noise=scipy.rand(50  2)*20  -3
  
#  Noisy_Data=numpy.concatenate((Data  noise))
#  size=20
  
  
#  fig  =  plt.figure()
#  ax1=fig.add_subplot(2  1  1)  #row    column    figure  number
#  ax2  =  fig.add_subplot(212)
  
#  ax1.scatter(Data[:  0]  Data[:  1]    alpha  =    0.5  )
#  ax1.scatter(noise[:  0]  noise[:  1]  color='red'    alpha  =    0.5)
#  ax2.scatter(noise[:  0]  noise[:  1]  color='red'    alpha  =    0.5)
  
  
Epsilon_min=1
Epsilon_max=2.0
MinumumPoints=4
Max_Points=5
result,cluster_len,Membership_value,Type=DBSCAN(Data,Epsilon_min,Epsilon_max,MinumumPoints,Max_Points)
PC = calculate_PC(len(Data),cluster_len,Membership_value)
FPI = calculate_FPI(len(Data),cluster_len,Membership_value)
F_measure  =  0
l  =  list(set(Output))
print(l)
for  i  in  range(len(l)):
    f_Precision,c1  =  F_Precision(Data,i,result,l,Output,Membership_value)
    f_recall,c2  =  F_recall(Data,i,result,l,Output,Membership_value)
    # count_cluster_point  =  Count_cluster(i,result)
    # count_class_point  =  Count_Class(i,Output)
    if  c2 !=  0  and  (f_recall+f_Precision)  !=0:
        F_measure  +=  (c1/c2)*((f_Precision*f_recall)/(f_recall+f_Precision))

#printed  numbers  are  cluster  numbers
# ,round(PC,2),round(FPI,2), F_measure
print  (round(PC,2),round(FPI,2),F_measure)
Out = Display(len(Data),Membership_value,Type,cluster_len)
for i in range(len(Out)):
    print(i,"::",Out[i])
#print  "Noisy_Data"
#print  Noisy_Data.shape
#print  Noisy_Data
  
#  for  i  in  range(len(result)):
#      ax2.scatter(Noisy_Data[i][0]  Noisy_Data[i][1]  color='yellow'    alpha  =    0.5)
        

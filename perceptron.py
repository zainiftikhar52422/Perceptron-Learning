import random
import pandas as pd
import sys
import os


print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))
arg=sys.argv[(sys.argv.index("-f"))+1]
print(arg)


def classify(fileName=arg):
    flowers=pd.read_csv(fileName)
    rows,columns=flowers.shape
    weight_o=round(random.uniform(-0.5,0.5), 3)
    x_o= -1
    alpha=0.001
    flage=True     # flage will be false when convergence reaches
    classified=1
    epoches=10000

    #print(rows,columns)
    '''for i in range(rows):
        if(flowers.loc[i,"species"]=="Iris-setosa"):
            flowers.loc[i,"species"]=1
        else:
            flowers.loc[i,"species"]=0'''
    weights=[round(random.uniform(-0.5,0.5), 3) for i in range(columns-1)]
    #print(weights)
    examps=list()
    for i in range(rows):
        tup=tuple(flowers.loc[i, :].values.tolist())
        examps.append(tup)
    columnName=list()
    columnName.append("epoch")
    for k in range(len(examps[0])-1):
        columnName.append("x"+str(k))
    columnName.append("Acctual")
    for k in range(len(weights)):
        columnName.append("w"+str(k))
    columnName.append("w_o")
    for k in range(len(weights)):
        columnName.append("updatedW"+str(k))
    columnName.append("updatedW_o")

    columnName.append("Perdicted")
    columnName.append("error")
    dataFrame=pd.DataFrame(columns= columnName)
        
    for j in range(epoches):
        flage=False
        for examp in examps:
            da=dict()
            da["epoch"]=j
            sums=0
            for i in range(len(weights)):
                da["x"+str(i)]=examp[i]
                da["w"+str(i)]=weights[i]
                sums=sums+examp[i]*weights[i]
            sums=sums+(weight_o*x_o)
            da["w_o"]=weight_o
            da["Acctual"]=examp[-1]
            
            if(sums>=0):      #if perceptoron excited
                output=1    
            else:
                output=0
            da["Perdicted"]=output
            da["error"]=examp[(len(weights))]-output
            if((output-examp[(len(weights))])!=0):    
                for i in range(len(weights)):
                    weights[i]=round(weights[i]+(alpha*(examp[(len(weights))]-output)*examp[i]),3)   # to round the out to 1 decimal point
                weight_o=round(weight_o+(alpha*(examp[(len(weights))]-output)*x_o),3)   # to round the out to 1 decimal point
                flage=True
            for i in range(len(weights)):
                da["updatedW"+str(i)]=weights[i]
            da["updatedW_o"]=weight_o
            #print(str(da))
            dataFrame = dataFrame.append(da, ignore_index=True)
        if(flage==False):
            print("completed In Epoches",j)
            break
    #os.popen("rm " + "log.csv")
    dataFrame.to_csv("log.csv",index=False)    
    return (weights,weight_o) 
def testModel(fileName,weights,weight_o):
    flowers_test=pd.read_csv(fileName)
    rows,columns=flowers_test.shape
    falseNegtive=0
    trueNegtive=0
    falsePostive=0
    truePostive=0
    x_o=-1
    examps=list()
    for i in range(rows):
        tup=tuple(flowers_test.loc[i, :].values.tolist())
        examps.append(tup)
    for examp in examps:
        sums=0
        for i in range(len(weights)):
            sums=sums+((examp[i]))*weights[i]
        sums=sums+(weight_o*x_o)    
        if(sums>=0):      #if perceptoron excited
            output=1    
        else:
            output=0
        if((output-examp[(len(weights))])!=0):    
            if(output==0):
                falseNegtive=falseNegtive+1
            elif(output==1):
                falsePostive=falsePostive+1
        elif(output==1):
            truePostive=truePostive+1
        elif(output==0):
            trueNegtive=trueNegtive+1
    print("Accuracy: ",((truePostive+trueNegtive)/(falseNegtive+falsePostive+trueNegtive+truePostive))*100,"%") 


if __name__ == '__main__': 
    for i in range(13):
        weights,weight_o=(classify())
        print(weights,weight_o)
        arg=arg
        testModel(arg,weights,weight_o)

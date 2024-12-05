import math
import random
from matplotlib import pyplot as plt
import numpy as npy
PI = 3.14159265358979323846
LEARNINGRATE = 1

#class van een layer: bevat de weights,biases en nodeswaardes van een rij nodes en de gewichten die binnen komen(de input layer wordt appart gedef aangezien hier niks binnenkomt)
class layer:
    def __init__(self,weights,bias,incomingNodes,outgoingNodes,nodeValues):
        self.weights = weights
        self.bias = bias
        self.incomingNodes = incomingNodes
        self.outgoingNodes = outgoingNodes
        self.nodeValues = nodeValues

#activatie functie, werkt
def sigmoid(x):
    if(x<-10000):
        return 0
    if(x>10000):
        return 1
    return  1/(1+npy.exp(-1*x))
def sigmoid_vector(arr):
    sigmoid_func = npy.vectorize(sigmoid)
    return sigmoid_func(arr)

#afgeleide activatiefunctie,werkt
def sigmoidDerivative(x):
    y = sigmoid(x)
    return y * (1 - y)


def logit(x):#inverse sigmoid
    if(x==1):
        return 10000
    if(x==0):
        return -10000
    return npy.log(x/(1-x))

#forward propegation: berekent de waardes van de nodes in de volgende layer,werkt
def forwardPropLayer(incomingLayer,outgoingLayer):
    print(outgoingLayer.outgoingNodes)
    print(incomingLayer.nodeValues)
    outgoingLayer.nodeValues = npy.dot(outgoingLayer.weights, incomingLayer.nodeValues) + outgoingLayer.bias
    if (outgoingLayer.outgoingNodes != 1):#bij de output line mag sigmoid niet toegepast worden, aangzien output als enigste 1 outgoing node heeft is dit een "goede" tijdelijke oplossing
            outgoingLayer.nodeValues = sigmoid_vector(outgoingLayer.nodeValues)
    return outgoingLayer


def forwardProp(network):#voert forward prop uit op een netwerk(lijst van lagen)
    
    for i in range(1,len(network)):
        network[i] = forwardPropLayer(network[i-1],network[i])
    return network


#bereken de costfunctie: voer forwardpropegation uit met de huidige weights & biases, bereken dan MSE,werkt
def cost(network,TrainingData, size):
    cost = 0
    npy.random.shuffle(TrainingData)
    
    for i in range(size):#TrainingData.shape[0]):
        TrainingDataComponent = TrainingData[i]
        network[0].nodeValues = npy.array([[TrainingDataComponent[0]],[TrainingDataComponent[1]]])
        network = forwardProp(network)
        cost += (TrainingDataComponent[2]-network[-1].nodeValues[i][0])**2

    cost = cost/size#TrainingData.shape[0]
    return cost

#1ste versie gradient descent: voor elke laag in het netwerk: bereken gradient numeriek met def afgeleide (f(x+h)-f(x))/h)

def gradientDescent(trainingsdata,network):
    originalCost = cost(network,trainingsdata,trainingsdata.shape[0])
    print("OriginalCost: ",end="")
    print(originalCost)
    h = 0.1

    
    for i in range(1,len(network)):
        gradientWeights = npy.zeros((network[i].outgoingNodes, network[i].incomingNodes))
        gradientBias = npy.zeros((network[i].outgoingNodes, 1))

        #bewerking is hier analoog aan bias maar dan in 2 dimensies
        for k in range(network[i].outgoingNodes):
            network[i].bias[k][0] += h #(x+h)
            Cost = cost(network,trainingsdata,trainingsdata.shape[0])#f(x+h) 
            gradientBias[k][0] = ((Cost-originalCost)/h) #voeg def afgeleide toe aan originalCost
            network[i].bias[k][0] -=h #trek h er terug vanaf zodat deze geen impact heeft op volgende berekeningen

            for l in range(network[i].incomingNodes):
                network[i].weights[k][l] +=h
                Cost = cost(network,trainingsdata,trainingsdata.shape[0])
                gradientWeights[k][l]=(Cost-originalCost)/h
                network[i].weights[k][l] -=h     
        #print(gradientWeights)
        #print(gradientBias)
        network[i].bias -= LEARNINGRATE * gradientBias
        network[i].weights -= LEARNINGRATE * gradientWeights    
    return network


#voor elk datapunt, forwardProp hiermee en voer dan gradientdescent uit
def Learn(network,trainingsdata):
    for k in range(1000):
        print("learn routine "+str(k))
        network = gradientDescent(trainingsdata,network)

def main():
    size = 10 #de grootte van de lagen in het midden

    trainingsdata = npy.array([[1,1,0],[1,0,1],[0,0,0],[0,1,1]])
    print(trainingsdata)
    layersize = [8, size, size, 4]
    network=[layer(0,0,0,8,[[0],[0],[0],[0],[0],[0],[0],[0]])]
    for i in range(1,len(layersize)):# aantal layers
        randweights = npy.array([[random.random() for i in range(network[i-1].outgoingNodes)] for _ in range(layersize[i])])
        randbias = npy.array([[random.random()] for _ in range(layersize[i])])
        network.append(layer(randweights, randbias, network[i-1].outgoingNodes, layersize[i], npy.zeros((layersize[i],1))))
    
    Learn(network,trainingsdata)
    print("netwerk getraind")
    network[0].nodeValues = npy.array([[trainingsdata[0][0]], [trainingsdata[0][1]],[trainingsdata[1][0]], [trainingsdata[1][1]],[trainingsdata[2][0]], [trainingsdata[2][1]],[trainingsdata[3][0]], [trainingsdata[3][1]]])
    y = forwardProp(network)[-1].nodeValues
    print(y)
    for i in range(trainingsdata.shape[0]):
        example = trainingsdata[i]
        print(str(example[0])+" and " +str(example[1])+" gives: "+str(y[i][0]))

if __name__ =="__main__":
    main()

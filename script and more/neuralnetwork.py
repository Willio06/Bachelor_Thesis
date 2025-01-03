import math
import random
from matplotlib import pyplot as plt
PI = 3.14159265358979323846
LEARNINGRATE = 0.01

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
    return  1/(1+math.exp(-1*x))

#afgeleide activatiefunctie,werkt
def sigmoidDerivative(x):
    y = sigmoid(x)
    return y * (1 - y)


#forward propegation: berekent de waardes van de nodes in de volgende layer,werkt
def forwardPropLayer(incomingLayer,outgoingLayer):
    for i in range(outgoingLayer.outgoingNodes):
        outgoingLayer.nodeValues[i] = outgoingLayer.bias[i]
        for j in range(outgoingLayer.incomingNodes):
            outgoingLayer.nodeValues[i] += incomingLayer.nodeValues[j]*outgoingLayer.weights[i][j]
        if (outgoingLayer.outgoingNodes != 1):#bij de output line mag sigmoid niet toegepast worden, aangzien output als enigste 1 outgoing node heeft is dit een "goede" tijdelijke oplossing
            outgoingLayer.nodeValues[i] = sigmoid(outgoingLayer.nodeValues[i])
    return outgoingLayer


def forwardProp(network):#voert forward prop uit op een netwerk(lijst van lagen)
    
    for i in range(1,len(network)):
        network[i] = forwardPropLayer(network[i-1],network[i])
    return network

#bereken de MSE van de laatste laag en de verwachte output,werkt
def MSE(TrainingData,outputLayer): 
    MSE = 0
    for i in range(outputLayer.outgoingNodes):
        eps = TrainingData[1] - outputLayer.nodeValues[i]
        MSE +=  (eps * eps)
    return MSE


#bereken de costfunctie: voer forwardpropegation uit met de huidige weights & biases, bereken dan MSE,werkt
def cost(network,TrainingDataComponent):
    network[0].nodeValues = [TrainingDataComponent[0]]
    network = forwardProp(network)
    Cost = MSE(TrainingDataComponent,network[-1])
    return Cost

#1ste versie gradient descent: voor elke laag in het netwerk: bereken gradient numeriek met def afgeleide (f(x+h)-f(x))/h)
def gradientDescent(trainingsdatacomponent,network):
    originalCost = cost(network,trainingsdatacomponent) 
    h = 0.001
    for i in range(1,len(network)):
        gradientWeights = []
        gradientBias = []
        for j in range(network[i].outgoingNodes):
            network[i].bias[j] += h #(x+h)
            Cost = cost(network,trainingsdatacomponent)#f(x+h) 
            gradientBias.append((Cost-originalCost)/h) #voeg def afgeleide toe aan originalCost
            network[i].bias[j] -=h#trek h er terug vanaf zodat deze geen impact heeft op volgende berekeningen

        #bewerking is hier analoog aan bias maar dan in 2 dimensies
        for k in range(network[i].outgoingNodes):
            gradientWeights.append([])
            for l in range(network[i].incomingNodes):
                network[i].weights[k][l] +=h
                Cost = cost(network,trainingsdatacomponent)
                gradientWeights[k].append((Cost-originalCost)/h)
                network[i].weights[k][l] -=h
        #print(gradientWeights)
        #print(gradientBias)
        for m in range(network[i].outgoingNodes):
            network[i].bias[m] -= LEARNINGRATE*gradientBias[m]
            for n in range(network[i].incomingNodes):
                network[i].weights[m][n] -= LEARNINGRATE*gradientWeights[m][n]
    return network
#def random dataset voor sinus
def sinustest(amountdata):
    trainingsdata = []
    for i in range(amountdata):
        x = random.uniform(-1*PI,PI)
        trainingsdata.append([x,math.sin(x)])
    return trainingsdata

#voor elk datapunt, forwardProp hiermee en voer dan gradientdescent uit
def Learn(network,trainingsdata):
    for i in range(len(trainingsdata)):
        network[0].nodeValues = [trainingsdata[i][0]]
        network = forwardProp(network)
        network = gradientDescent(trainingsdata[i],network)


    
def main():
    size = 20 #de grootte van de laag in het midden
    trainingsize = 1000
    trainingsdata = sinustest(trainingsize)
    #print(trainingsdata)
    layersize = [1, size, size, size]
    network=[layer(0,0,0,1,[0])]
    for i in range(1,len(layersize)):# aantal layers
        randweights = [[random.random() for i in range(network[i-1].outgoingNodes)] for _ in range(layersize[i])]
        randbias = [random.random() for _ in range(layersize[i])]
        network.append(layer(randweights, randbias, network[i-1].outgoingNodes, layersize[i], [0.]*layersize[i]))
    
    Learn(network,trainingsdata)
    print("netwerk getraind")

    xvals = []
    yvals = []
    #hier gebruiken we het netwerk om functiewaardes te benaderen: y = netwerk(x)
    for i in range(trainingsize):
        #print(i)
        x = random.uniform(-1*PI,PI)
        xvals.append(x)
        network[0].nodeValues = [x]
        y = forwardProp(network)[-1].nodeValues[0]
        yvals.append(y)
    #print(xvals)
    #print(yvals)
    plt.scatter(xvals,yvals)
    plt.show()


        

if __name__ =="__main__":
    main()

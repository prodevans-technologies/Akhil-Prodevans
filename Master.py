'''Created and Developed by Akhil.K'''
'''Find me at blitzkrieg.akhil@gmail.com'''
'''Encoded input where 1 - access.log, 2 - anaconda.log, 3 - boot.log, 4 - cron.log, 5 - error.log, 6 - firewall.log'''
'''7 - messages, 8 - yum.log, 9 - Unknown'''
#Capsule for NNPrototype.py
feed=[]
def encoder(log):
    if log !='':
        if 'GET' in log:
            feed.append([1,0,0,0,0,0,0,0,0])
            print(1)
        elif 'anaconda' in log:
            feed.append([0,1,0,0,0,0,0,0,0])
        elif '[0m]' in log:
            feed.append([0,0,1,0,0,0,0,0,0])
        elif 'cron' in log.lower():
            feed.append([0,0,0,1,0,0,0,0,0])
        elif 'AH0' in log:
            feed.append([0,0,0,0,1,0,0,0,0])
        elif 'ERROR:' in log:
            feed.append([0,0,0,0,0,1,0,0,0])
        elif 'ambari-server' in log and 'cron' not in log.lower():
            feed.append([0,0,0,0,0,0,1,0,0])
        elif 'Updated:' in log or 'Installed:' in log or 'Erased:' in log:
            feed.append([0,0,0,0,0,0,0,1,0])
        else:
            feed.append([0,0,0,0,0,0,0,0,1])
def capsule():
    log=input("Enter the log ")
    while log:
        encoder(log)
        log=input("Enter the log ")
        encoder(log)

import random
import numpy as np
import math

#Standard bias
bias = [random.random() for i in range(3)]
#Bias weights
biasw = [[random.random() for i in range(4)],[random.random() for i in range(4)],[random.random() for i in range(2)]]
lr = 0.3
network=[]
nodes=[]

def sigmoid(x):
    # Calculates value of sigmoid function
    ret=[]
    for i in x:
        result=math.exp(i)
        result=np.array(result)
        result=1+result
        result=1/result
        ret.append(result)
    return ret

def initialize_network(n_inputs=9, n_hidden1=4,n_hidden2=4, n_outputs=2):
    # Random initialization of weights
    # Nested list contains weights previous layer to every node in present layer
    network = list()
    hidden_layer1 = [[random.random() for i in range(n_inputs)] for i in range(n_hidden1)]
    network.append(hidden_layer1)
    hidden_layer2 = [[random.random() for i in range(n_hidden1)] for i in range(n_hidden2)]
    network.append(hidden_layer2)
    output_layer = [[random.random() for i in range(n_hidden2)] for i in range(n_outputs)]
    network.append(output_layer)
    return network

def forward_propogation(network, inputs,n_inputs=1, n_hidden1=4,n_hidden2=4, n_outputs=2):
    # Calculate neuron activation for an input, store it in nodes list and forward propogate
    nodes = list()
    hidden_layer1 = [(np.dot(network[0][i],inputs)+bias[0]*biasw[0][i]) for i in range(n_hidden1)]
    hidden_layer1 = np.tanh(hidden_layer1)
    nodes.append(hidden_layer1)
    hidden_layer2 = [(np.dot(network[1][i],hidden_layer1)+bias[1]*biasw[1][i]) for i in range(n_hidden2)]
    hidden_layer2 = np.tanh(hidden_layer2)
    nodes.append(hidden_layer2)
    output_layer = [(np.dot(network[2][i],hidden_layer2)+bias[2]*biasw[2][i]) for i in range(n_outputs)]
    output_layer = sigmoid(output_layer)
    nodes.append(output_layer)
    return nodes

def back_propogation(network,nodes,expected_output,inputs):
    # Using calculus, updates weights according to error generated
    # Update weights between output layer and hidden layer 2
    network[2][0] = np.array(network[2][0]) - lr*(expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*(nodes[1])
    #biasw[2][0] = biasw[2][0] - lr*(expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*bias[2]
    network[2][1] = np.array(network[2][1]) - lr*(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*(nodes[1])
    #biasw[2][1] = biasw[2][1] - lr*(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*bias[2]
    # Update weights between hidden layer 2 and hidden layer 1
    network[1][0] = np.array(network[1][0]) - lr*(1-(np.tanh(np.sum(network[1][0])))**2)*(np.array(nodes[0]))*((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*(network[2][0][0])+(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*(network[2][1][0]))
    #biasw[1][0] = biasw[1][0] - lr*(1-(np.tanh(np.sum(network[1][0])))**2)*(bias[1])*((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*(network[2][0][0])+(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*(network[2][1][0]))
    network[1][1] = np.array(network[1][1]) - lr*(1-(np.tanh(np.sum(network[1][1])))**2)*(np.array(nodes[0]))*((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*(network[2][0][1])+(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*(network[2][1][1]))
    #biasw[1][1] = biasw[1][1] - lr*(1-(np.tanh(np.sum(network[1][1])))**2)*(bias[1])*((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*(network[2][0][1])+(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*(network[2][1][1]))
    network[1][2] = np.array(network[1][2]) - lr*(1-(np.tanh(np.sum(network[1][2])))**2)*(np.array(nodes[0]))*((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*(network[2][0][2])+(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*(network[2][1][2]))
    #biasw[1][2] = biasw[1][2] - lr*(1-(np.tanh(np.sum(network[1][2])))**2)*(bias[1])*((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*(network[2][0][2])+(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*(network[2][1][2]))
    network[1][3] = np.array(network[1][3]) - lr*(1-(np.tanh(np.sum(network[1][3])))**2)*(np.array(nodes[0]))*((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*(network[2][0][3])+(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*(network[2][1][3]))
    #biasw[1][3] = biasw[1][3] - lr*(1-(np.tanh(np.sum(network[1][3])))**2)*(bias[1])*((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*(network[2][0][3])+(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*(network[2][1][3]))
    # Update weights between hidden layer 1 and input layer
    network[0][0] = np.array(network[0][0]) - lr*np.array(inputs)*(1-(np.tanh(np.sum(network[0][0])))**2)*(((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*((network[1][0][0]*network[2][0][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][0]*network[2][0][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][0]*network[2][0][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][0]*network[2][0][3]*(1-(np.tanh(np.sum(network[1][3])))**2))))+((expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*((network[1][0][0]*network[2][1][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][0]*network[2][1][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][0]*network[2][1][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][0]*network[2][1][3]*(1-(np.tanh(np.sum(network[1][3])))**2)))))
    #biasw[0][0] = biasw[0][0] - lr*bias[0]*(1-(np.tanh(np.sum(network[0][0])))**2)*(((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*((network[1][0][0]*network[2][0][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][0]*network[2][0][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][0]*network[2][0][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][0]*network[2][0][3]*(1-(np.tanh(np.sum(network[1][3])))**2))))+((expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*((network[1][0][0]*network[2][1][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][0]*network[2][1][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][0]*network[2][1][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][0]*network[2][1][3]*(1-(np.tanh(np.sum(network[1][3])))**2)))))
    network[0][1] = np.array(network[0][1]) - lr*np.array(inputs)*(1-(np.tanh(np.sum(network[0][1])))**2)*(((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*((network[1][0][1]*network[2][0][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][1]*network[2][0][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][1]*network[2][0][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][1]*network[2][0][3]*(1-(np.tanh(np.sum(network[1][3])))**2))))+((expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*((network[1][0][1]*network[2][1][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][1]*network[2][1][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][1]*network[2][1][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][1]*network[2][1][3]*(1-(np.tanh(np.sum(network[1][3])))**2)))))
    #biasw[0][1] = biasw[0][1] - lr*bias[0]*(1-(np.tanh(np.sum(network[0][1])))**2)*(((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*((network[1][0][1]*network[2][0][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][1]*network[2][0][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][1]*network[2][0][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][1]*network[2][0][3]*(1-(np.tanh(np.sum(network[1][3])))**2))))+((expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*((network[1][0][1]*network[2][1][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][1]*network[2][1][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][1]*network[2][1][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][1]*network[2][1][3]*(1-(np.tanh(np.sum(network[1][3])))**2)))))
    network[0][2] = np.array(network[0][2]) - lr*np.array(inputs)*(1-(np.tanh(np.sum(network[0][2])))**2)*(((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*((network[1][0][2]*network[2][0][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][2]*network[2][0][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][2]*network[2][0][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][2]*network[2][0][3]*(1-(np.tanh(np.sum(network[1][3])))**2))))+((expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*((network[1][0][2]*network[2][1][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][2]*network[2][1][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][2]*network[2][1][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][2]*network[2][1][3]*(1-(np.tanh(np.sum(network[1][3])))**2)))))
    #biasw[0][2] = biasw[0][2] - lr*bias[0]*(1-(np.tanh(np.sum(network[0][2])))**2)*(((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*((network[1][0][2]*network[2][0][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][2]*network[2][0][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][2]*network[2][0][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][2]*network[2][0][3]*(1-(np.tanh(np.sum(network[1][3])))**2))))+((expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*((network[1][0][2]*network[2][1][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][2]*network[2][1][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][2]*network[2][1][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][2]*network[2][1][3]*(1-(np.tanh(np.sum(network[1][3])))**2)))))
    network[0][3] = np.array(network[0][3]) - lr*np.array(inputs)*(1-(np.tanh(np.sum(network[0][3])))**2)*(((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*((network[1][0][3]*network[2][0][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][3]*network[2][0][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][3]*network[2][0][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][3]*network[2][0][3]*(1-(np.tanh(np.sum(network[1][3])))**2))))+((expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*((network[1][0][3]*network[2][1][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][3]*network[2][1][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][3]*network[2][1][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][3]*network[2][1][3]*(1-(np.tanh(np.sum(network[1][3])))**2)))))
    #biasw[0][3] = biasw[0][3] - lr*bias[0]*(1-(np.tanh(np.sum(network[0][3])))**2)*(((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*((network[1][0][3]*network[2][0][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][3]*network[2][0][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][3]*network[2][0][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][3]*network[2][0][3]*(1-(np.tanh(np.sum(network[1][3])))**2))))+((expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*((network[1][0][3]*network[2][1][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][3]*network[2][1][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][3]*network[2][1][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][3]*network[2][1][3]*(1-(np.tanh(np.sum(network[1][3])))**2)))))
    return network

expected_output=[]   #Edit this

def epochs(epochs):
    #Training phase
    inputs=[]
    network = initialize_network()
    for i in range(epochs):
        inputs.append(feed[i])
        nodes = forward_propogation(network,inputs[i])
        network = back_propogation(network,nodes,expected_output[i],inputs[i])
    #Testing phase
    #Enter something to test the neural network
#Enter number of epochs
capsule()
epochs(len(feed))
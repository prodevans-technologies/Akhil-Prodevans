import random
import numpy as np
import math

#Standard bias
bias = [random.random() for i in range(3)]
#Bias weights
biasw = [[random.random() for i in range(4)],[random.random() for i in range(4)],[random.random() for i in range(2)]]
lr = 0.3

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

def initialize_network(n_inputs=1, n_hidden1=4,n_hidden2=4, n_outputs=2):
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
    hidden_layer1 = [np.dot(network[0][i],inputs)+bias[0]*biasw[0][i] for i in range(n_hidden1)]
    hidden_layer1 = np.tanh(hidden_layer1)
    nodes.append(hidden_layer1)
    hidden_layer2 = [np.dot(network[1][i],hidden_layer1)+bias[1]*biasw[1][i] for i in range(n_hidden2)]
    hidden_layer2 = np.tanh(hidden_layer2)
    nodes.append(hidden_layer2)
    output_layer = [np.dot(network[2][i],hidden_layer2)+bias[2]*biasw[2][i] for i in range(n_outputs)]
    output_layer = sigmoid(output_layer)
    nodes.append(output_layer)
    return nodes

def back_propogation(network,nodes,expected_output,inputs):
    # Using calculus, updates weights according to error generated
    expo = expected_output
    # Update weights between output layer and hidden layer 2
    network[2][0] = np.array(network[2][0]) - lr*(expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*np.array(nodes[1])
    biasw[2][0] = biasw[2][0] - lr*(expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*bias[2]
    network[2][1] = np.array(network[2][1]) - lr*(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*np.array(nodes[1])
    biasw[2][1] = biasw[2][1] - lr*(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*bias[2]
    # Update weights between hidden layer 2 and hidden layer 1
    network[1][0] = np.array(network[1][0]) - lr*(1-(np.tanh(np.sum(network[1][0])))**2)*(np.array(nodes[0]))*((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*(network[2][0][0])+(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*(network[2][1][0]))
    biasw[1][0] = biasw[1][0] - lr*(1-(np.tanh(np.sum(network[1][0])))**2)*(bias[1])*((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*(network[2][0][0])+(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*(network[2][1][0]))
    network[1][1] = np.array(network[1][1]) - lr*(1-(np.tanh(np.sum(network[1][1])))**2)*(np.array(nodes[0]))*((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*(network[2][0][1])+(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*(network[2][1][1]))
    biasw[1][1] = biasw[1][1] - lr*(1-(np.tanh(np.sum(network[1][1])))**2)*(bias[1])*((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*(network[2][0][1])+(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*(network[2][1][1]))
    network[1][2] = np.array(network[1][2]) - lr*(1-(np.tanh(np.sum(network[1][2])))**2)*(np.array(nodes[0]))*((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*(network[2][0][2])+(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*(network[2][1][2]))
    biasw[1][2] = biasw[1][2] - lr*(1-(np.tanh(np.sum(network[1][2])))**2)*(bias[1])*((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*(network[2][0][2])+(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*(network[2][1][2]))
    network[1][3] = np.array(network[1][3]) - lr*(1-(np.tanh(np.sum(network[1][3])))**2)*(np.array(nodes[0]))*((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*(network[2][0][3])+(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*(network[2][1][3]))
    biasw[1][3] = biasw[1][3] - lr*(1-(np.tanh(np.sum(network[1][3])))**2)*(bias[1])*((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*(network[2][0][3])+(expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*(network[2][1][3]))
    # Update weights between hidden layer 1 and input layer
    network[0][0] = np.array(network[0][0]) - lr*inputs[0]*(1-(np.tanh(np.sum(network[0][0])))**2)*(((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*((network[1][0][0]*network[2][0][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][0]*network[2][0][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][0]*network[2][0][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][0]*network[2][0][3]*(1-(np.tanh(np.sum(network[1][3])))**2))))+((expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*((network[1][0][0]*network[2][1][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][0]*network[2][1][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][0]*network[2][1][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][0]*network[2][1][3]*(1-(np.tanh(np.sum(network[1][3])))**2)))))
    biasw[0][0] = biasw[0][0] - lr*bias[0]*(1-(np.tanh(np.sum(network[0][0])))**2)*(((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*((network[1][0][0]*network[2][0][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][0]*network[2][0][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][0]*network[2][0][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][0]*network[2][0][3]*(1-(np.tanh(np.sum(network[1][3])))**2))))+((expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*((network[1][0][0]*network[2][1][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][0]*network[2][1][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][0]*network[2][1][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][0]*network[2][1][3]*(1-(np.tanh(np.sum(network[1][3])))**2)))))
    network[0][1] = np.array(network[0][1]) - lr*inputs[0]*(1-(np.tanh(np.sum(network[0][1])))**2)*(((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*((network[1][0][1]*network[2][0][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][1]*network[2][0][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][1]*network[2][0][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][1]*network[2][0][3]*(1-(np.tanh(np.sum(network[1][3])))**2))))+((expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*((network[1][0][1]*network[2][1][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][1]*network[2][1][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][1]*network[2][1][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][1]*network[2][1][3]*(1-(np.tanh(np.sum(network[1][3])))**2)))))
    biasw[0][1] = biasw[0][1] - lr*bias[0]*(1-(np.tanh(np.sum(network[0][1])))**2)*(((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*((network[1][0][1]*network[2][0][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][1]*network[2][0][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][1]*network[2][0][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][1]*network[2][0][3]*(1-(np.tanh(np.sum(network[1][3])))**2))))+((expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*((network[1][0][1]*network[2][1][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][1]*network[2][1][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][1]*network[2][1][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][1]*network[2][1][3]*(1-(np.tanh(np.sum(network[1][3])))**2)))))
    network[0][2] = np.array(network[0][2]) - lr*inputs[0]*(1-(np.tanh(np.sum(network[0][2])))**2)*(((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*((network[1][0][2]*network[2][0][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][2]*network[2][0][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][2]*network[2][0][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][2]*network[2][0][3]*(1-(np.tanh(np.sum(network[1][3])))**2))))+((expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*((network[1][0][2]*network[2][1][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][2]*network[2][1][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][2]*network[2][1][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][2]*network[2][1][3]*(1-(np.tanh(np.sum(network[1][3])))**2)))))
    biasw[0][2] = biasw[0][2] - lr*bias[0]*(1-(np.tanh(np.sum(network[0][2])))**2)*(((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*((network[1][0][2]*network[2][0][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][2]*network[2][0][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][2]*network[2][0][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][2]*network[2][0][3]*(1-(np.tanh(np.sum(network[1][3])))**2))))+((expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*((network[1][0][2]*network[2][1][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][2]*network[2][1][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][2]*network[2][1][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][2]*network[2][1][3]*(1-(np.tanh(np.sum(network[1][3])))**2)))))
    network[0][3] = np.array(network[0][3]) - lr*inputs[0]*(1-(np.tanh(np.sum(network[0][3])))**2)*(((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*((network[1][0][3]*network[2][0][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][3]*network[2][0][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][3]*network[2][0][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][3]*network[2][0][3]*(1-(np.tanh(np.sum(network[1][3])))**2))))+((expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*((network[1][0][3]*network[2][1][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][3]*network[2][1][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][3]*network[2][1][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][3]*network[2][1][3]*(1-(np.tanh(np.sum(network[1][3])))**2)))))
    biasw[0][3] = biasw[0][3] - lr*bias[0]*(1-(np.tanh(np.sum(network[0][3])))**2)*(((expected_output[0]-nodes[2][0])*(nodes[2][0])*(1-nodes[2][0])*((network[1][0][3]*network[2][0][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][3]*network[2][0][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][3]*network[2][0][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][3]*network[2][0][3]*(1-(np.tanh(np.sum(network[1][3])))**2))))+((expected_output[1]-nodes[2][1])*(nodes[2][1])*(1-nodes[2][1])*((network[1][0][3]*network[2][1][0]*(1-(np.tanh(np.sum(network[1][0])))**2))+(network[1][1][3]*network[2][1][1]*(1-(np.tanh(np.sum(network[1][1])))**2))+(network[1][2][3]*network[2][1][2]*(1-(np.tanh(np.sum(network[1][2])))**2))+(network[1][3][3]*network[2][1][3]*(1-(np.tanh(np.sum(network[1][3])))**2)))))
    return network

n=int(input("Enter the number of inputs "))
inputs=[]
for i in range(n):
    inputs.append(int(input("Enter input ")))
expected_output=[0,1]   #Edit this later

network = initialize_network()
nodes = forward_propogation(network,inputs)
updated_network = back_propogation(network,nodes,expected_output,inputs)

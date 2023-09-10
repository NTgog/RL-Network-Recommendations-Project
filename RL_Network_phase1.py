import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt 

# Environment parameters

K = 50  # Number of content items
u_min = 0.2  # Threshold for content relevance
C = int(0.2 * K)  # Number of cached items
#C = 2
# User model parameters
N = 2 # Number of recommended items
q = 0.2  # Probability of ending the viewing session
alpha = 0.8  # Probability of selecting a recommended item
#tradeoff_factor=0.4
# Generate random relevance values
U = np.random.rand(K, K)
np.fill_diagonal(U, 0)  # Set diagonal elements to 0
# U = np.array([[0., 0.09709217, 0.95697935, 0.76421269, 0.79379138],
#               [0.85679266, 0., 0.73115609, 0.97025111, 0.00706508],
#               [0.38327773, 0.27582305, 0., 0.40938946, 0.70918518],
#               [0.27415892, 0.89691232, 0.47103534, 0., 0.97776446],
#               [0.06699551, 0.96500574, 0.00547615, 0.74654658, 0.]])
# U = np.array([[0.0, 0.8, 0.3, 0.6],
#               [0.8, 0.0, 0.7, 0.2],
#               [0.3, 0.1, 0.0, 0.2],
#               [0.6, 0.4, 0.2, 0.0]])
print(U)
#vector to denote the cost of each state. 1 for non-cached, 0 for cached
Cost = [1]*(K-C) +[0]*C   
#Cost = [1,0,1,0]
random.shuffle(Cost)
print(Cost)

Tmax = 10000

#create action set as the set of every possible combination of N=2 states 
action_set = []
for i in range(K):
    for j in range(i+1,K):
        a = (i, j)
        action_set.append(a)

num_of_actions = len(action_set)

action_table = [[] for _ in range(K)]

for i in range(K):
    for a in action_set:
        if i not in a:
            action_table[i].append(a)
num_of_actions = len(action_table[0])
print(num_of_actions)
print (action_table)
a = random.choice(action_table[2])
s_prime = int(np.random.choice(a))
print(a)
print(s_prime)

########################################################################              Policy Iteration                  ###########################################################################################

def all_recommendations_are_relevant(recommendations,s):
    """
    function to check whether everey recommended state in the racommendation batch is 
    relevant to s

    arguments:
    recommendations (tuple of ints): recommendation batch for state s
    s (int): current state

    returns:
    True: if all recommendation are relevant to s
    False: otherwise
    """
    for u in recommendations:
        if U[s][int(u)]<u_min:
            return False
    return True

def get_next_states(s, a): 
    """
    function to generate all possible next states alongiside their respective probabilities and rewards
    given a state s and an action a

    arguments:
    s (int):  current state
    a (tuple of ints): recommendation batch for state s

    returns:
    next_states (list of tuples):a list of tuples, where each tuple denotes the next state alongiside its respective probability and reward
    """
    next_states = []
    if(all_recommendations_are_relevant(a, s)): #determine the probabilities for next given that all recommendation are relevant
        for s_prime in range(K):
                if s_prime in a:
                    prob = (alpha/N + (1-alpha)/K) #probability to choose a video from the recommedation batch 
                    reward = 1-Cost[s_prime] #define reward to be 1 for cached and 0 for uncached video
                    next_states.append((prob,s_prime,reward))
                else:  
                    prob = (1-alpha)/K #probability to choose a video outside the recommedation batch 
                    reward = 1-Cost[s_prime]
                    next_states.append((prob,s_prime,reward))
            
    else: #determine the probabilities for next given that at least one recommendation is irrelevant
        for s_prime in range(K):
                
                prob = 1/K 
                reward = 1-Cost[s_prime]
                next_states.append((prob,s_prime,reward))
                
    return next_states
       
def policy_evaluation(pi, gamma = 1.0, epsilon = 1e-10):  
    """
    Function to evaluate a given policy

    arguments:
    pi (list of tuples): policy to be evaluated
    gamma (float): discounting factor
    epsilon (float): approximation tolerance

    returns:
    V (vector of floats): vector of value function of each state calculated using bellman equation
    """

    prev_V = np.zeros(K) # use as "cost-to-go", i.e. for V(s')
    while True: #performing iterations
        V = np.zeros(K) # current value function to be learnerd
        for s in range(K):  # do for every state
            for prob, next_state, reward in get_next_states(s, pi[s]):  # calculate one Bellman step --> i.e., sum over all probabilities of transitions and reward for that state, the action suggested by the (fixed) policy, the reward earned (dictated by the model), and the cost-to-go from the next state (which is also decided by the model)
                V[s] += prob * (reward + gamma * prev_V[next_state] )
        if np.max(np.abs(prev_V - V)) < epsilon: #check if the new V estimate is close enough to the previous one; 
            break # if yes, finish loop
        prev_V = V.copy() #freeze the new values (to be used as the next V(s'))
    return V





def policy_improvement(V, gamma=1.0):  
    """
    Function to greedily improve policy

    arguments:
    V (vector of floats): V (vector of floats): vector of value function of each state calculated using bellman equation
    gamma (float): discounting factor
    
    returns:
    new_pi (list of tuples): new policy
    """
    Q = np.zeros((K, num_of_actions), dtype=np.float64) #create a Q value array
    for s in range(K):        # for every state in the environment/model
        for i,a in enumerate(action_table[s]):  # and for every action in that state
            for prob, next_state, reward in get_next_states(s, a):  #evaluate the action value based on the model and Value function given (which corresponds to the previous policy that we are trying to improve) 
                Q[s][i] += prob * (reward + gamma * V[next_state] )
#     new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]  # this basically creates the new (improved) policy by choosing at each state s the action a that has the highest Q value (based on the Q array we just calculated)
    new_pi = np.zeros((K, N), dtype=np.int16)
    for s in range(K):
        
        best_action = np.argmax(Q[s])
           
        new_pi[s] = action_table[s][best_action]
    
    return new_pi


def policy_iteration( gamma = 1.0, epsilon = 1e-10):
    """
    Function to converge to the optimal policy by performing iterative avaluation and improvments until convergence

    arguments:
    gamma (float): discounting factor
    epsilon (float): approximation tolerance

    returns:
    pi (list of tuples): optimal policy
    V (vector of floats): vector of value function of each state calculated using bellman equation
    """
    t = 0
    # random_actions = np.random.choice(tuple(P[0].keys()), len(P))     # start with random actions for each state  
    # pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]     # and define your initial policy pi_0 based on these action (remember, we are passing policies around as python "functions", hence the need for this second line)
    
    pi = np.random.randint(0, K, size=(K, N))  # Random initial policy
    #print(np.sum(R,axis=1))

    while True:
        old_pi = pi.copy()  #keep the old policy to compare with new
        V = policy_evaluation(pi,gamma,epsilon)   #evaluate latest policy --> you receive its converged value function

        pi = policy_improvement(V,gamma)          #get a better policy using the value function of the previous one just calculated 
        
        t += 1

        if compare_two_policies(pi,old_pi) or t>10: # you have converged to the optimal policy if the "improved" policy is exactly the same as in the previous step
            break
    print('converged after %d iterations' %t) #keep track of the number of (outer) iterations to converge
   
    return V,pi
    
def compare_two_policies(pi,old_pi):
    """
    Function to compare if two policies are identical

    arguments:
    pi: current policy
    old: previous policy

    returns:
    True if they are identical
    Flase otherwise
    """
    for k in range(K):
        for n in range(N):
            if (pi[k][n]!=old_pi[k][n]):
                return False

    return True


V_opt,P_opt1 = policy_iteration(1-q,0.001)   #just example of calling the various new functions we created.



print("#############################")
#print(V_opt)
print("#############################")
print(P_opt1)

########################################################################              Q-Learning                  ###########################################################################################

def Q_learning(gamma, epsilon,learning_rate):
    """
    Function to perform Q learning algorithm.

    arguments:
    gamma (float): discounting factor
    epsilon (float): epslilon greedy probability
    learning_rate (float): learning rate

    returns:
    Q (matrix K x num_of_actions):  martix of state action value function calculated using bellman equation
    """
    Q = np.zeros((K,num_of_actions)) 
    prev_Q = np.zeros((K,num_of_actions))
    t = 0
    while True:
        s = np.random.randint(K) #random initial state
        while True:
            if np.random.uniform() < epsilon:  # Explore if e(t) times
                
                a_idx = np.random.randint(num_of_actions) #choose random action
                    
            else:  # Exploit 1-e(t) times
                
                a_idx = np.argmax(Q[s]) #choose greedily the action with highest Q value
            a = action_table[s][a_idx]  

            if (all_recommendations_are_relevant(a,s)):
            
                if np.random.uniform() < alpha:  # If all recommended items are relevant
                    s_prime = int(np.random.choice(a))  # Pick a random item from relevant recommended items
                else:  # If at least one recommended item is not relevant
                    s_prime = np.random.randint(K)  # Pick a random item
            else:
                s_prime = np.random.randint(K)  # Pick a random item
        
            if np.random.uniform() < q: #if user opt to terminate session
                target = (1/2 - Cost[s_prime])
                Q[s][a_idx] = prev_Q[s][a_idx] + learning_rate * ( target - prev_Q[s][a_idx] )
                break
            else:
                target = (1/2 - Cost[s_prime]) - Cost[s_prime] + gamma*np.max(prev_Q[s_prime])
            Q[s][a_idx] = prev_Q[s][a_idx] + learning_rate * ( target - prev_Q[s][a_idx] )
            
            s = s_prime
        t+=1
        epsilon = (t+1)**(-1/3)*(num_of_actions*math.log(t+1))**(1/3)
        #epsilon = 0.1
        #learning_rate = learning_rate*(1/t)**(1/2)
        
        #if (np.max(np.abs(prev_Q - Q)) < delta and t>1000*K) or 
        if t > 2000*K: #check if the new V estimate is close enough to the previous one;
            break # if yes, finish loop
        prev_Q = Q.copy()
    print(t)
    return Q


pi_Q_learning  =  np.zeros((K, N), dtype=np.int16)

Q = Q_learning(1-q,1,0.01)

for s in range(K):
    #Q[s][s] = float('-inf')
    action = np.argmax(Q[s])
    pi_Q_learning[s] = action_table[s][action]

print(pi_Q_learning)

########################################################################              SIMULATION                  ###########################################################################################

def simulate_session(policy, max_steps=1000):
    """
    Simulate a viewing session following a given policy

    arguments:
    policy to be simulated

    returns:
    total cost of the session
    
    """
    s = np.random.randint(K)  # random initial
    cost_total = Cost[s]  
    for _ in range(max_steps):
        if np.random.uniform() < q:  # The user decides to quit
            break

        if (all_recommendations_are_relevant(policy[s],s)):
            
            if np.random.uniform() < alpha:  # If all recommended items are relevant
                s_prime = int(np.random.choice(policy[s]))  # Pick a random item from relevant recommended items
            else:  # If at least one recommended item is not relevant
                s_prime = np.random.randint(K)  # Pick a random item
        else:
            s_prime = np.random.randint(K)  # Pick a random item
        
        s=s_prime
        cost_total += Cost[s]  # Add the cost of the picked item
    return cost_total

def simulation(policy):
    """
    function to run multiple sessions
    """
    total_cost = 0
    num_of_episodes=50000
    for _ in range(num_of_episodes):
        total_cost  += simulate_session(policy)

    print(total_cost/num_of_episodes)

print(U)
print(Cost)
#print(P_opt1)
#print(pi_Q_learning)

print("average cost for Policy iteration:")
simulation(P_opt1)
#simulation(P_opt2)
print("average cost for Q Learning:")
simulation(pi_Q_learning)
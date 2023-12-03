# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 23:12:59 2022

@author: Korisnik
"""




#%% Libraries

import matplotlib.pyplot as plt
import numpy as np
import random as r

#%% Function that determines whether the game is over

def game_over(state = tuple):
    x = (1,1,1)
    o = (-1,-1,-1)
    
    win_x = ( state[0:3]==x ) or ( state[3:6]==x ) or ( state[6:9]==x ) or ( (state[0],state[3],state[6])==x ) or ( (state[1],state[4],state[7])==x ) or ( (state[2],state[5],state[8])==x ) or ( (state[0],state[4],state[8])==x ) or ( (state[2],state[4],state[6])==x )
    win_o = ( state[0:3]==o ) or ( state[3:6]==o ) or ( state[6:9]==o ) or ( (state[0],state[3],state[6])==o ) or ( (state[1],state[4],state[7])==o ) or ( (state[2],state[5],state[8])==o ) or ( (state[0],state[4],state[8])==o ) or ( (state[2],state[4],state[6])==o )

    if win_x:
        return('x')
    elif win_o:
        return('o')
    elif not( 0 in state ):
        return('DRAW')
    else:
        return(False)
    
state = (1,1,-1,
         -1,-1,1,
         1,1,-1)

print(game_over(state))

#%% Function that determines how player move based on Q table

def player_move(state = tuple, Q = dict, explore = int, symbol = int):
    moves = Q[state]
    if r.random() > explore:
        idx_max = max(moves, key=moves.get)
        state = tuple(state[i] if i!=idx_max else symbol for i in range(9))
        return(state, idx_max)
    else:
        idx = r.choice(list(moves))
        state = tuple(state[i] if i!=idx else symbol for i in range(9))
        return(state, idx)
        

#%% Initialization of parameters

    
board = (0,0,0,
         0,0,0,
         0,0,0) # 0=empty, 1=x, -1=o

Q1 = {}
Q1[board] = {idx: 0 for idx in range(9) if board[idx]==0}
Q2 = {}

alpha = 0.1
gamma = 1
explore = 1
explore_decay = 0.95
explore_decay_time = 500
N = 10**5 # number of games to train on

win_reward = 1
lose_reward = -1
draw_reward = 0

#%% Update Q's rules

def Qmax(Qstate):
    try:
        return( max( Qstate.values() ) )
    except:
        return(0)

def updateQ1(past_state, state, action, reward = 0):
    global Q1
    
    if not( state in Q1 ):
        Q1[state] = {idx: 0 for idx in range(9) if state[idx]==0}
    
    Q1[past_state][action] = (1-alpha)*Q1[past_state][action] + alpha*(reward + gamma * Qmax(Q1[state]) )
    
    
def updateQ2(past_state, state, action, reward = 0):
    global Q2
    
    if not( state in Q2 ):
        Q2[state] = {idx: 0 for idx in range(9) if state[idx]==0}
    
    Q2[past_state][action] = (1-alpha)*Q2[past_state][action] + alpha*(reward + gamma * Qmax(Q2[state]) )
    
#%% Playing\Training

Game_Record = [0]*N


for game in range(1, N+1):
    
    board2, action1 = player_move(board, Q1, explore, 1)
    try:
        board1, action2 = player_move(board2, Q2, explore, -1)
    except:
        Q2[board2] = {idx: 0 for idx in range(9) if board2[idx]==0}
        board1, action2 = player_move(board2, Q2, explore, -1)
    
    # Update Q1
    updateQ1(board, board1, action1)
    
    
    for turn in range(5): # There are in total max of 6 turns for x and 5 for o but one is already played above
        board2_past = board2
        board1_past = board1
        
        # Player X moves
        board2, action1 = player_move(board1, Q1, explore, 1)

        game_stage = game_over(board2)
        
        if game_stage:
            if game_stage=='x':
                Game_Record[game-1] = ('x')
                updateQ2(board2_past, board2, action2, lose_reward)
                updateQ1(board1_past, board2, action1, win_reward)
            else:
                Game_Record[game-1] = ('DRAW')
                updateQ2(board2_past, board2, action2, draw_reward)
                updateQ1(board1_past, board2, action1, draw_reward)
            break
            
        # Update Q2 in case game is not over
        updateQ2(board2_past, board2, action2)
        
        # Player O moves
        board1, action2 = player_move(board2, Q2, explore, -1)
        
        game_stage = game_over(board1)
        
        if game_stage:
            if game_stage=='o':
                Game_Record[game-1] = ('o')
                updateQ2(board2_past, board1, action2, win_reward)
                updateQ1(board1_past, board1, action1, lose_reward)
            else:
                Game_Record[game-1] = ('DRAW')
                updateQ2(board2_past, board1, action2, draw_reward)
                updateQ1(board1_past, board1, action1, draw_reward)
            break
        

        # Update Q1 in case game is not over
        updateQ1(board1_past, board1, action1)
        
        
    
    if game%explore_decay_time==0:
        explore *= explore_decay
        
#%%

GAMES = [i for i in range(N//100)]
X_WINS = []
O_WINS = []
DRAWS = []


for i in GAMES:
    x_wins = 0
    o_wins = 0
    draws = 0
    for j in range(i*100, (i+1)*100):
        if Game_Record[j] == 'x':
            x_wins += 1
        elif Game_Record[j] == 'o':
            o_wins += 1
        else:
            draws += 1
    X_WINS.append(x_wins/100)
    O_WINS.append(o_wins/100)
    DRAWS.append(draws/100)
    

GAMES = [i*100 for i in GAMES]

plt.plot(GAMES, X_WINS, label = 'X win')
plt.plot(GAMES, O_WINS, label = 'O win')
plt.plot(GAMES, DRAWS, label = 'Draw')
plt.xlabel('Number of games')
plt.ylabel('Percentage')
plt.legend()
plt.show()

#%% Lengths of Q1 and Q2

q1 = len(Q1)
q2 = len(Q2)


#%% Converting Q tables to csv files

player1 = np.empty( (6, q1*3) )
player2 = np.empty( (6, q2*3) )

player1[:] = np.NaN
player2[:] = np.NaN

position = 0

for state in Q1:
    state_vec = np.asarray(state)
    player1[0][position:position+3] = state_vec[0:3]
    player1[1][position:position+3] = state_vec[3:6]
    player1[2][position:position+3] = state_vec[6:9]
    for move in Q1[state]:
        player1[3+move//3][position+move%3] = Q1[state][move]
    position += 3
    
    
position = 0

for state in Q2:
    state_vec = np.asarray(state)
    player2[0][position:position+3] = state_vec[0:3]
    player2[1][position:position+3] = state_vec[3:6]
    player2[2][position:position+3] = state_vec[6:9]
    for move in Q2[state]:
        player2[3+move//3][position+move%3] = Q2[state][move]
    position += 3
    
    


np.savetxt(r"C:\Users\Korisnik\OneDrive\Desktop\Neural Networks Chalmers-Gothenburg\HW3\Tic-Tac-Toe\player1.csv", player1, delimiter=',')
np.savetxt(r"C:\Users\Korisnik\OneDrive\Desktop\Neural Networks Chalmers-Gothenburg\HW3\Tic-Tac-Toe\player2.csv", player2, delimiter=',')










        
    
    












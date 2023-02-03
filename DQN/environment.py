import numpy as np
import pygame as pg

class Snake_environment: 
    def __init__(self): 
        self.row = 8                    # Nombre de lignes
        self.col = 8                    # Nombre de colonnes
        self.snake_initial = 2
        self.time_penalty = -3          # Penalité de temps
        self.death_penalty = -100       # Pénalité de mort
        self.apple_reward = 200         # Gain

        self.snake_position = []
        
        #Grille de jeu : 0 = rien, 0.5 = serpent, 1 = pomme
        self.map =  np.zeros((self.row, self.col))

        for i in range(self.snake_initial):  #Taille initiale d'un snake : self.snake_initial 
            row_init = int(self.row/2)+i
            col_init = int(self.col/2)
            self.snake_position+=[(row_init,col_init)]
            self.map[row_init,col_init] = 0.5

        self.apple_position = self.new_apple()

        self.collected = False
        self.last_action = 0

    def new_apple(self):
        new_row = np.random.randint(0,self.row)
        new_col = np.random.randint(0,self.col)

        while self.map[new_row,new_col]==0.5 : # i.e. le serpent est dessus
            new_row = np.random.randint(0,self.row)
            new_col = np.random.randint(0,self.col)

        self.map[new_row,new_col] = 1  # On place la pomme sur la grille
        return(new_row,new_col)

    def snake_move(self,next_row,next_col,apple_collected):
        self.snake_position.insert(0,(next_row,next_col))

        if not apple_collected :
            self.snake_position.pop()
        # Sinon le serpent grandit de 1, donc on ne change rien

        elif apple_collected :
            self.collected = True

        # On redéfini la grille :
        self.map = np.zeros((self.row,self.col))

        for i in range(len(self.snake_position)):  #len(self.snake_position)= taille du serpent + 1
            pos_row = self.snake_position[i][0]
            pos_col = self.snake_position[i][1]
            self.map[pos_row,pos_col]=0.5

        apple_pos_row = self.apple_position[0]
        apple_pos_col = self.apple_position[1]
        self.map[apple_pos_row,apple_pos_col]= 1

    def step(self,action):
        '''
        Notre convention : 
        ACTION_UP = 0
        ACTION_DOWN = 1
        ACTION_RIGHT = 2
        ACTION_LEFT = 3
        '''
        #de base, on réinitialise ça : (en premier si jamais il rencontre une pomme ensuite)
        self.collected = False

        # Il faut d'abord vérifier que la nouvelle action est réalisable, dans le cas contraire, le serpent avance de 1 dans sa direction
        # Les différents mouvements irréalisable sont :
        #    - aller à gauche après être aller à droite => l'action réalisée sera alors aller à droite
        #    - aller en haut après être aller en bas => l'action réalisée sera alors aller en bas
        #    - aller à droite après être aller à gauche => l'action réalisée sera alors aller à gauche
        #    - aller en bas après être aller en haut => l'action réalisée sera alors aller en haut
        if   action == 3 and self.last_action == 2 :
            action = 2
        elif action == 0 and self.last_action == 1 :
            action = 1
        elif action == 2 and self.last_action == 3 :
            action = 3
        elif action == 1 and self.last_action == 0 :
            action = 0

        # On effectue l'action souhaitée, regarde le Rt associé et effectue une étape
        pos_row = self.snake_position[0][0]
        pos_col = self.snake_position[0][1]

        if action == 0 :
            next_row = pos_row-1
            next_col = pos_col
            
            #Le serpent perd s'il rencontre un mur ou lui-même:
            #Grille de jeu : 0 = rien, 0.5 = serpent, 1 = pomme
            if next_row < 0 :
                game_over = True
                reward = self.death_penalty
            else:
                if self.map[next_row,next_col] == 0.5:
                    game_over = True
                    reward = self.death_penalty
                if self.map[next_row,next_col] == 1:
                    game_over = False
                    reward = self.apple_reward
                    self.snake_move(next_row,next_col, apple_collected=True)
                if self.map[next_row,next_col] == 0:
                    game_over = False
                    reward = self.time_penalty
                    self.snake_move(next_row,next_col, apple_collected = False)

        elif action == 1 :
            next_row = pos_row+1
            next_col = pos_col
            if next_row > self.row -1 :
                game_over = True
                reward = self.death_penalty
            else:
                if self.map[next_row,next_col] == 0.5:
                    game_over = True
                    reward = self.death_penalty
                if self.map[next_row,next_col] == 1:
                    game_over = False
                    reward = self.apple_reward
                    self.snake_move(next_row,next_col, apple_collected=True)
                if self.map[next_row,next_col] == 0:
                    game_over = False
                    reward = self.time_penalty
                    self.snake_move(next_row,next_col, apple_collected = False)

        elif action == 2 :
            next_row = pos_row
            next_col = pos_col+1
            if next_col > self.col -1 :
                game_over = True
                reward = self.death_penalty
            else:
                if self.map[next_row,next_col] == 0.5:
                    game_over = True
                    reward = self.death_penalty
                if self.map[next_row,next_col] == 1:
                    game_over = False
                    reward = self.apple_reward
                    self.snake_move(next_row,next_col, apple_collected=True)
                if self.map[next_row,next_col] == 0:
                    game_over = False
                    reward = self.time_penalty
                    self.snake_move(next_row,next_col, apple_collected = False)
        elif action == 3 :
            next_row = pos_row
            next_col = pos_col-1
            if next_col < 0 :
                game_over = True
                reward = self.death_penalty
            else:
                if self.map[next_row,next_col] == 0.5:
                    game_over = True
                    reward = self.death_penalty
                if self.map[next_row,next_col] == 1:
                    game_over = False
                    reward = self.apple_reward
                    self.snake_move(next_row,next_col, apple_collected=True)
                if self.map[next_row,next_col] == 0:
                    game_over = False
                    reward = self.time_penalty
                    self.snake_move(next_row,next_col, apple_collected = False)

        # On reset les paramètres du jeu correctement (seulement maintenant, car aurait pu changer du choix initial)
        self.last_action = action

        return(self.map,reward,game_over)

    def reset(self):
        self.map = np.zeros((self.row,self.col))
        self.snake_position = []

        for i in range(self.snake_initial):
            row_init = int(self.row/2)+i
            col_init = int(self.col/2)

            self.snake_position+=[(row_init,col_init)]

            self.map[row_init,col_init] = 0.5

        apple_pos_row = self.apple_position[0]
        apple_pos_col = self.apple_position[1]
        self.map[apple_pos_row,apple_pos_col]= 1
    
        self.last_move = 0
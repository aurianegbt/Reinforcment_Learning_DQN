import numpy as np

class DDQN(object):
    def __init__(self,memory_capacity=100,gamma=0.9):
        self.memory = []
        self.memory_capacity = memory_capacity
        self.discount = gamma

    def save_exp(self,exp,game_over):   #La variable exp=[St,At,Rt,St+1]
        self.memory.append([exp,game_over])
        if len(self.memory) > self.memory_capacity :
            self.memory.pop(0)

    def get_batch(self,model1,model2, size_batch):
        # dans la mémoire : M[i]=[[St, At, Rt, St+1], game_over]

        n_memory = len(self.memory)
        n_actions = model1.output_shape[-1] #méthode de Seuqential() 

        size_limit = min(n_memory,size_batch)

        states = np.zeros((size_limit,self.memory[0][0][0].shape[1],self.memory[0][0][0].shape[2],self.memory[0][0][0].shape[3]))  # on créer un tableau 4D pour extraire size_limit vecteur 3D car les éléments dans memory sont de la forme [[St,At,Rt,St+1],done] Donc memory[0][0][0]=St et on décide que :
        # St est un vecteur 3D : une dimension pour les lignes de la grilles, une pour les colonnes des grilles
        # St retient l'état de la grille des n dernières actions (d'où la troisième dimension de St 
        targets = np.zeros((size_limit,n_actions))

        batch_set=np.random.randint(0,n_memory,size=size_limit)
        for i,ind in enumerate(batch_set):
            St, At, Rt, Stp1 = self.memory[ind][0]
            game_over = self.memory[ind][1]
            states[i] = St
             # Prédiction pour trouver l'action qui donne le plus grand Q-value avec le modèle cible
            max_action = np.argmax(model1.predict(Stp1)[0])
            # Prédiction du Q-value correspondant à l'action sélectionnée avec le modèle courant
            Q = model2.predict(Stp1)[0][max_action]
            if game_over:
                targets[i,At] = Rt
            else :
                targets[i,At] = Rt + self.discount * Q

        return(states,targets)
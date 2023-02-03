from environment import Snake_environment
from cnn import neural_kernel
from DDQN import DDQN
import numpy as np
import matplotlib.pyplot as plt

''' Définition des paramètres '''
capacity_memory = 50000
size_batch = 32
learningRate = 0.0001
gamma = 0.9
n_last_action = 4  #nbr de dernières actions sauvegardé pour la 3e dimension de St évoque dans les notebook précédent 

eps = 1           #Probabilité de faire un mouvement aléatoire (permet de tester de nouveaux états -> évite les minimum locaux)
down_step=0.0002  #Plus on progresse, moins les mouvements sont aléatoires
min_eps = 0.05

filepathToSave1 = 'model1.h5'
filepathToSave2 = 'model2.h5'

''' Initialise tout '''
env = Snake_environment()
neural_network=neural_kernel((env.row,env.col,n_last_action),learningRate)
model1 = neural_network.model
model2 = neural_network.model
ddqn=DDQN(capacity_memory,gamma)

def reset_state():
    state = np.zeros((1,env.row,env.col,n_last_action))
    for i in range(n_last_action):
        state[:,:,:,i]=env.map
    return(state) 


episode = 0
apple_collected = 0              # Pommes collectées pendant 1 épisodes
max_apple_collected = 0          # Max des pommes collectés pendant 1 seul épisode
scores = []                      # Liste des scores (i.e. pommes récupérées) -> pour plus lisible, faire graphe par 10/20/50 ou 100 (selon le nbr fait...)
max_scores = []                  # Listes des scores maximaux

while True:
    env.reset()
    St, Stp1 = reset_state(),reset_state()
    episode += 1
    game_over = False

    print('Episode n°'+str(episode)+' débuté...')
    # Let's go !
    
    while not game_over: # -> début d'une partie 
        if np.random.rand() <= eps :    #Action aléatoire ou non
            action = np.random.randint(0,4)
        else :
            Q = model2.predict(St)[0] 
            action = np.argmax(Q)

        # On met à jour en fonction de l'action décidé :
        state_aux,rew,game_over = env.step(action)  #décide la prochaine étape et la sauvegarde pour le moment dans la variable auxiliaire state

        state_aux = np.reshape(state_aux, (1, env.row, env.col, 1))
        Stp1 = np.append(Stp1, state_aux, axis = 3)
        Stp1 = np.delete(Stp1,0,axis=3)

        # Et on sauvegarde l'expérience :
        ddqn.save_exp([St,action,rew,Stp1],game_over)
        states,targets = ddqn.get_batch(model1,model2,size_batch)
        model1.train_on_batch(states,targets)
        
        #Pour éviter les divergends entre les modèles, on set les poids du model2 sur ceux du model1 :
        model2.set_weights(model1.get_weights())

        if env.collected:
            apple_collected +=1

        St=Stp1

        #On met à jour les récompenses à la fin de la partie :

    if apple_collected > max_apple_collected :
        max_apple_collected = apple_collected
        model1.save(filepathToSave1)
        model2.save(filepathToSave2)
    max_scores.append(max_apple_collected)

    # On récupère le score à chaque partie :
    scores.append(apple_collected)


    apple_collected =0
    # Pour nos différents graphs :
    # A mesure que l'IA progresse, on réduit la stochasticité dans l'algorithme
    if eps > min_eps:
        eps = eps- down_step

    with open("scores.txt", "w") as output:
        output.write('scores ='+str(scores)+'\n')
        output.write('max_scores = '+str(max_scores)+'\n')

    print('episode n°'+str(episode)+ ' terminé, avec '+str(scores[-1])+' pomme(s) collectée(s).\n')


# Implémentation des algorithmes DQN et DDQN.
## Introduction 
L'apprentissage par renforcement (ou _Reinforcment Learning_, RL) consiste en un agent qui, à force d'entraînement et donc à force d'acquérir de l'expérience, décide des actions à prendre selon l'état de l'environnement où il évolue. Une récompense, positive ou négative, lui est alors retourné selon l'action prise et son effet sur l'environnement. L'objectif de l'agent sera alors d'optimiser les décisions qu'il prend pour maximiser la somme de ses récompenses.

Un premier cadre populaire d'apprentissage par renforcement est l'univers des jeux vidéos. L'environnement serait alors le jeu dans lequel l'agent évolue; les actions qu'il peut prendre sont les différentes possibilités de mouvements dans le jeu, contraintes par les règles de celui-ci; et les récompenses peuvent être associé à la réussite d'une mission, d'un niveau, ou à un Game Over, une pénalité, etc

## Motivations

``[Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)'' est un article publié en 2017 qui présente une amélioration de l'algorithme DQN (Deep $Q$-Network) en combinant plusieurs améliorations qui ont été proposées précédemment dans la littérature. Les auteurs ont combiné $7$ améliorations différentes, déjà existente, pour créer l'algorithme Rainbow, qui montre des résultats bien meilleurs que l'algorithme original. Les améliorations utilisées sont le Double $Q$-Learning, _Prioritized Replay_, le _Dueling Network_, _Multi-Step Learning_, _Distributional Reinforment Learninf_, et _Noisy Nets_. 

Dans le cadre de l'article, l'algorithme DQN est entraîner et tester sur les jeux Atari. Les jeux Atari sont une série de jeux vidéos développés et commercialisés par Atari, Inc. dans les années 1970 et 1980. Ils ont été initialement publiés sur les consoles du même nom. Ces jeux ont été très populaires à l'époque et ont contribué à populariser les consoles de salon. Plusieurs de ces jeux, tels que PacMan, Space Invaders et Asteroids, sont devenus des icônes de la culture populaire. Ces jeux sont un cadres simples et très accessibles dans lequel faire du Reinforcment Learning. L'article utilise pour cela les packages gymnasium qui implémente complètement les environnements de ses jeux et regroupe un grand nombres de méthodes (e.g. _n\_actions_, _.action\_space_, etc.).

Notre objectif ici est de reprendre les idées données par l'article en présentant, en amont, les notions et concepts nécessaire à sa compréhension. Nous avons pu tester deux algorithmes d'apprentissage évoqués par l'article dont nous comparerons les performances. Faute d'une installation réussi de _gymnasium_, nous proposons un entraînement et des tests sur un environnement inspiré du jeu ``Snake''. 


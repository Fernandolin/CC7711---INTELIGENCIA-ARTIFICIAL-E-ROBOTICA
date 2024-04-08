import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

arrayLoss = []

print('Carregando Arquivo de teste')
arquivo = np.load('teste5.npy')

for z in range(1,11):
    
    x = arquivo[0]
    y = np.ravel(arquivo[1])

    
    regr = MLPRegressor(hidden_layer_sizes=(700,300),
                        max_iter=1000,
                        activation='relu', #{'identity', 'logistic', 'tanh', 'relu'},
                        solver='adam',
                        learning_rate = 'adaptive',
                        n_iter_no_change=500)
    
    #print('Treinaz'ndo RNA')
    regr = regr.fit(x,y)
    
    
    
    #print('Preditor')
    y_est = regr.predict(x)
    
    
    
    plt.figure(figsize=[14,7])
    
    #plot curso original
    plt.subplot(1,3,1)
    plt.plot(x,y)
    
    plt.title(f"Gráfico do teste {z}")
    #plot aprendizagem
    plt.subplot(1,3,2)
    plt.plot(regr.loss_curve_)
    
    #plot regressor
    plt.subplot(1,3,3)
    plt.plot(x,y,linewidth=1,color='yellow')
    plt.plot(x,y_est,linewidth=2)
    
    print("Teste ", z)
    print("Best loss: ", regr.best_loss_, end="\n")
    
    arrayLoss.append(regr.best_loss_)
    plt.show()

media = np.mean(arrayLoss)
desvio = np.std(arrayLoss)
print(f"A média do BestLoss é : {media:.6f}")
print(f"O desvio do BestLoss é : {desvio:.6f}")

menor_valor = min(arrayLoss)
posicao = arrayLoss.index(menor_valor)
teste = posicao + 1 
print(f"Menor valor: {menor_valor}, Posição: {teste}")


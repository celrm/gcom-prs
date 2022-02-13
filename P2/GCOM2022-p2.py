"""
Práctica 2
"""
import numpy as np
import pandas as pd
from collections import Counter

def distribution_from_file(path):
    contents = open(path, 'r',encoding="utf8").read()
    frecuency_dict = Counter(contents)    
    distribution = pd.DataFrame(frecuency_dict.most_common()).iloc[::-1].reset_index(drop=True)
    distribution.columns = ['state','probab']
    distribution.probab = distribution.probab/float(sum(distribution.probab))
    return distribution

def huffman_branch(distr):
    state_new = distr.state[0] + distr.state[1]
    probab_new = distr.probab[0] + distr.probab[1]
    code = [{distr.state[0]: 0, distr.state[1]: 1}]
    distr.loc[len(distr.index)] = [state_new, probab_new]
    distr = distr.drop([0,1])
    distr.sort_values(by='probab', ascending=True, inplace=True,ignore_index=True)
    return (distr, np.array([code]))

def huffman_tree(distr):
    tree = np.array([])
    while len(distr) > 1:
        distr, code = huffman_branch(distr)
        tree = np.concatenate((tree, code), axis=None)
        print(distr)
    return(tree)

distr_eng = distribution_from_file('GCOM2022_pract2_auxiliar_eng.txt')
distr_esp = distribution_from_file('GCOM2022_pract2_auxiliar_esp.txt')
tree_eng = huffman_tree(distr_eng)
tree_eng[0].items()
tree_eng[0].values()

#Buscar cada estado dentro de cada uno de los dos items
list(tree_eng[0].items())[0][0] ## Esto proporciona un '0'
list(tree_eng[0].items())[1][0] ## Esto proporciona un '1'

"""
por linea, separar cada letra

tabla frec
ordenamos
huffman
0 1
codificamos cada letra
longitud media

eficiencia con binario utf8 (comparar long media)
descifrar
"""

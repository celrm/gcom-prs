'''
Práctica 2
'''
import numpy as np
import pandas as pd
from collections import Counter

# Me permite hacer el apartado iii) en las mismas condiciones que la plantilla.
# Columna extra para desambiguar el orden de los elems de igual probabilidad.
col = 'TpCmAkq;bfy?WBv’\n-S,lr.dnIcwgiuaoseth '
col = {l: i for i, l in enumerate(col)}

# Dado el nombre de un fichero, devuelve un Dataframe con columnas
# 'state' (letras) y 'probabs' (frecuencias normalizadas) ordenado
# de menor a mayor frecuencia. Se desambigua por col.
def distribution_from_file(path):
    contents = open(path, 'r', encoding='utf8').read()
    frecs = Counter(contents)  # cuento la frecuencia de cada letra
    distr = pd.DataFrame(frecs.most_common(), columns=['state', 'probab'])
    distr.probab = distr.probab / float(sum(distr.probab))  # normalizo probabs
    distr['s'] = [col.get(l, 0) for l in distr.state] # iii)
    distr.sort_values(['probab', 's'], ascending=True, ignore_index=True, inplace=True)
    del distr['s'] # iii)
    return distr


# Realiza una iteración del algoritmo. Fusiona filas 0 y 1,
# introduce el nuevo elemento, y reordena de nuevo.
def huffman_branch(distr):
    code = {
        'value0': distr.state[0],
        'value1': distr.state[1],
        'child0': distr.child[0],  # para recorrer el árbol luego
        'child1': distr.child[1],  # para recorrer el árbol luego
    }
    distr.loc[len(distr.index)] = {
        'state': distr.state[0] + distr.state[1],  # concat states
        'probab': distr.probab[0] + distr.probab[1],  # sum probabs
        'child': len(distr.index) - 2,  # para recorrer el árbol luego
    }
    distr = distr.drop([0, 1])
    distr.sort_values('probab', ascending=True, ignore_index=True, inplace=True)
    return distr, code


# Devuelve el árbol de Huffman de una tabla de letras y probabilidades.
# Una lista que empieza en el 0 y se recorre con cada índice childx.
def huffman_tree(distr):
    distribution = distr.copy()
    distribution['child'] = None  # si son hojas se quedarán así
    tree = []
    while len(distribution) > 1:
        distribution, code = huffman_branch(distribution)
        tree = [code] + tree
    return tree


# Dado un árbol de Huffman, devuelve un diccionario letra->código.
def codif_table(tree):
    codification = {}
    for code in tree:
        for l in code['value0']:
            codification[l] = codification.get(l, '') + '0'
        for l in code['value1']:
            codification[l] = codification.get(l, '') + '1'
    return codification


# Para el apartado ii)
# Dado un diccionario letra->código, devuelve la codificación de una palabra.
def codification(codif, word):
    return ''.join([codif[l] for l in word])


# Para el apartado iii)
# Dado un árbol de Huffman y una palabra codificada, devuelve su original.
# El árbol es una lista cuya raíz es el primer elemento.
# Cada elem: Valor (0),  Valor (1), Índice del hijo (0) e Índice del hijo (1).
def decodification(tree, code):
    word = ''
    i = 0
    for c in code:
        if not tree[i]['child' + c]:
            word += tree[i]['value' + c]
            i = 0
        else:
            i = tree[i]['child' + c]
    return word


# Longitud media: sum( wi |ci| ) con wi la probabilidad normalizada y ci el código de cada letra
def get_L(distr, codif):
    return sum(distr.probab * [len(codif[s]) for s in distr.state])


# Entropía
def get_H(distr):
    return -sum(distr.probab * np.log2(distr.probab))


''' Apartado i) '''
print('-' * 10)

distr_eng = distribution_from_file('GCOM2022_pract2_auxiliar_eng.txt')
tree_eng = huffman_tree(distr_eng)
codif_eng = codif_table(tree_eng)
L_eng = get_L(distr_eng, codif_eng)
H_eng = get_H(distr_eng)

print('Inglés:\t\tH =', H_eng, '≤ L =', L_eng, '< H+1 =', H_eng + 1)

distr_esp = distribution_from_file('GCOM2022_pract2_auxiliar_esp.txt')
tree_esp = huffman_tree(distr_esp)
codif_esp = codif_table(tree_esp)
L_esp = get_L(distr_esp, codif_esp)
H_esp = get_H(distr_esp)

print('Español:\tH =', H_esp, '≤ L =', L_esp, '< H+1 =', H_esp + 1)


''' Apartado ii) '''
print('-' * 10)

word = 'medieval'

# Creamos la tabla de codificación binaria para el conjunto de todas las letras de los textos.
# La tabla no hace falta para saber la longitud de la palabra codificada, pues será
# su longitud sin codificar (8) * la longitud en binario de (len(Alfabeto)-1)
all_letters = set(distr_eng.state).union(set(distr_esp.state))
size_bin = len(bin(len(all_letters) - 1)[2:])
codif_bin = {l: bin(i)[2:].zfill(size_bin) for i, l in enumerate(all_letters)}

# Resultados:
result_bin = codification(codif_bin, word)
result_eng = codification(codif_eng, word)
result_esp = codification(codif_esp, word)

print('Codificación de la palabra:', word)
print('En binario (hay', len(all_letters), 'letras):', len(word), '*', size_bin, '=', len(word) * size_bin, result_bin)
print('Huffman inglés:\t\t\t', len(result_eng), result_eng)
print('Huffman español:\t\t', len(result_esp), result_esp)

print("Comprobación:",decodification(tree_eng,result_eng),decodification(tree_esp,result_esp))

''' Apartado iii) '''
print('-' * 10)

code = '10111101101110110111011111'
word = decodification(tree_eng, code)
print('Decodificación del código:', code)
print(word)
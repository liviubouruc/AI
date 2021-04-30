import numpy as np
import copy
import math

class NodParcurgere:
    def __init__(self, info, parinte, g, f):
        # self.id = id  # este indicele din vectorul de noduri
        self.info = info
        self.parinte = parinte  # parintele din arborele de parcurgere
        self.f = f
        self.g = g

    def obtineDrum(self):
        l = [self.info]
        nod = self
        while nod.parinte is not None:
            l.insert(0, nod.parinte.info)
            nod = nod.parinte
        return l

    def afisDrum(self):  # returneaza si lungimea drumului
        l = self.obtineDrum()
        print(("->").join([str(x) for x in l]))
        return len(l)

    def contineInDrum(self, infoNodNou):
        nodDrum = self
        while nodDrum is not None:
            if (infoNodNou == nodDrum.info):
                return True
            nodDrum = nodDrum.parinte

        return False

    def __repr__(self):
        sir = ""
        sir += self.info + "("
        sir += "id = {}, ".format(self.id)
        sir += "drum="
        drum = self.obtineDrum()
        sir += ("->").join(drum)
        sir += " cost:{})".format(self.cost)
        return (sir)


class Graph:  # graful problemei
    def __init__(self, start):
        self.start = start

    def indiceNod(self, n):
        return self.noduri.index(n)

    # va genera succesorii sub forma de noduri in arborele de parcurgere
    def genereazaSuccesori(self, nodCurent):
        listaSuccesori = []

        stive_start = nodCurent.info

        for i in range(len(stive_start)): # pt fiecare stiva i nevida
            if len(stive_start[i]) == 0:
                continue
            for j in range(len(stive_start)): # pt fiecare stiva j nevida != i
                if i == j:
                    continue
                stive_next = copy.deepcopy(stive_start)
                stive_next[j].append(stive_next[i][-1]) # punem pe stiva j varful de pe stiva i
                cost = stive_next[i].pop()  # scoatem de pe stiva i

                if not nodCurent.contineInDrum(stive_next): # daca nu am vizitat starea adaugam succesorul
                  listaSuccesori.append((stive_next, self.calculeaza_h(stive_next), nodCurent.g+1))

        return listaSuccesori

    def __repr__(self):
        sir = ""
        for (k, v) in self.__dict__.items():
            sir += "{} = {}\n".format(k, v)
        return (sir)

    def calculeaza_h(self, nod_info):
        s = 0
        for i in range(len(nod_info)):
            s += len(nod_info[i])
        return s // len(nod_info)

##############################################################################################
#                                 Initializare problema                                      #
##############################################################################################

# pozitia i din vector
start = [[1], [2, 3], [3]]
gr = Graph(start)


def in_list(nod_info, lista):
    for nod in lista:
        if nod_info == nod.info:
            return nod
    return None


def insert(node, lista):
    idx = 0
    while idx < len(lista) - 1 and (node.f > lista[idx].f or (node.f == lista[idx].f and node.g < lista[idx].g)):
        idx += 1
    lista.insert(idx, node)

def testare_scop(blocuri):
    for i in range(len(blocuri) - 1):
        for j in range(i + 1, len(blocuri)):
            if len(blocuri[i]) == 0 or len(blocuri[j]) == 0:
                continue
            if len(blocuri[j]) != len(blocuri[i]):
                return False
    return True

def a_star():
    # de completat
    opened = [NodParcurgere(start, None, 0, None)]
    closed = []

    continua = True

    while continua and len(opened) > 0:
        current_node = opened.pop(0)
        closed.append(current_node)

        if testare_scop(current_node.info):
            current_node.afisDrum()
            continua = False

        succesori = gr.genereazaSuccesori(current_node)
        for nod in succesori:
            info, h, g = nod
            node_open = in_list(info, opened)
            node_parc = NodParcurgere(info, current_node, g, g + h)
            if node_open is not None:
                if node_open.f > g + h:
                    opened.remove(node_open)
                    insert(node_parc, opened)
                continue
            node_closed = in_list(info, closed)
            if node_closed is not None:
                if node_closed.f > g + h:
                    closed.remove(node_closed)
                    insert(node_parc, opened)
                continue
            insert(node_parc, opened)

    if len(opened) == 0:
        print("Nu exista drum!")


if __name__ == '__main__':
    a_star()
import copy


class NodParcurgere:
    def __init__(self, info, parinte, g, f):
        self.info = info
        self.parinte = parinte  # parintele din arborele de parcurgere
        self.g = g
        self.f = f

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


class Graph:  # graful problemei
    def __init__(self, start, final):
        self.start = start
        self.final = final

    # va genera succesorii sub forma de noduri in arborele de parcurgere
    def genereazaSuccesori(self, nodCurent):
        listaSuccesori = []

        config = nodCurent.info
        posI = -1
        posJ = -1
        for i in range(len(config)):
            for j in range(len(config)):
                if config[i][j] == 0:
                    posI = i
                    posJ = j

        for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            if posI + dx < 0 or posI + dx >= 3 or posJ + dy < 0 or posJ + dy >= 3:
                continue

            newI = posI + dx
            newJ = posJ + dy

            new_config = copy.deepcopy(config)
            new_config[newI][newJ], new_config[posI][posJ] = new_config[posI][posJ], new_config[newI][newJ]

            listaSuccesori.append((new_config, self.calculeaza_h(config), nodCurent.g+1))

        return listaSuccesori

    def calculeaza_h(self, nod_info):
        diff = 0
        for i in range(len(nod_info)):
            for j in range(len(nod_info)):
                if nod_info[i][j] != self.final[i][j]:
                    diff += 1
        return diff

##############################################################################################
#                                 Initializare problema                                      #
##############################################################################################


#start = [[5, 7, 2], [8, 0, 6], [3, 4, 1]]
#final = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
start = [[1, 2, 0], [4, 5, 3], [7, 8, 6]]
final = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
gr = Graph(start, final)


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


def a_star():
    opened = [NodParcurgere(start, None, 0, 0)]
    closed = []

    continua = True

    while continua and len(opened) > 0:
        current_node = opened.pop(0)
        closed.append(current_node)

        if current_node.info == final:
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
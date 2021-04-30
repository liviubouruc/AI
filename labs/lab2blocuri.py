from copy import deepcopy

class NodParcurgere:
    def __init__(self, info, cost, parinte):
        # self.id = id  # este indicele din vectorul de noduri
        self.info = info
        self.parinte = parinte  # parintele din arborele de parcurgere
        self.cost = cost

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
    def genereazaSuccesori(self, nodCurent, c):
        listaSuccesori = []

        stive_start = nodCurent.info

        for i in range(len(stive_start)): # pt fiecare stiva i nevida
            if len(stive_start[i]) == 0:
                continue
            for j in range(len(stive_start)): # pt fiecare stiva j nevida != i
                if i == j:
                    continue
                stive_next = deepcopy(stive_start)
                stive_next[j].append(stive_next[i][-1]) # punem pe stiva j varful de pe stiva i
                cost = stive_next[i].pop()  # scoatem de pe stiva i

                if not nodCurent.contineInDrum(stive_next): # daca nu am vizitat starea adaugam succesorul
                  listaSuccesori.append(NodParcurgere(stive_next, cost + nodCurent.cost, nodCurent))

        return listaSuccesori

    def __repr__(self):
        sir = ""
        for (k, v) in self.__dict__.items():
            sir += "{} = {}\n".format(k, v)
        return (sir)

start = [[1], [2, 3], [3]]
nrSolutiiCautate = 5
gr = Graph(start)


def testare_scop(blocuri):
    for i in range(len(blocuri) - 1):
        for j in range(i + 1, len(blocuri)):
            if len(blocuri[i]) == 0 or len(blocuri[j]) == 0:
                continue
            if len(blocuri[j]) != len(blocuri[i]):
                return False
    return True


def uniform_cost(gr):
    global nrSolutiiCautate
    c = [NodParcurgere(start, 0, None)]
    continua = True
    while (len(c) > 0 and continua):
        nodCurent = c.pop(0)
        # print("Processing node ", nodCurent.info)

        if testare_scop(nodCurent.info):
            nodCurent.afisDrum()
            nrSolutiiCautate -= 1
            if nrSolutiiCautate == 0:
                continua = False

        lSuccesori = gr.genereazaSuccesori(nodCurent, c)
        for s in lSuccesori:
            i = 0
            gasit_loc = False
            for i in range(len(c)):
                # diferenta e ca ordonez dupa f
                if c[i].cost >= s.cost:
                    gasit_loc = True
                    break
            if gasit_loc:
                c.insert(i, s)
            else:
                c.append(s)

if __name__ == '__main__':
  print("Got here")
  uniform_cost(gr)
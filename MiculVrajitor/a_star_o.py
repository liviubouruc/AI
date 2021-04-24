import time


class NodParcurgere:
    def __init__(self, info, parinte, g, f):
        self.info = info  # informatii despre stare
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

    def afisDrum(self, cost, timp, noduri_generate, nr_max_n, fout):  # returneaza si lungimea drumului
        l = self.obtineDrum()

        fout.write(
            "Pas 0). Incepe drumul cu cizme de culoarea " + l[0][2] + " din locatia " + str(l[0][:2]) + ". Incaltat: " +
            l[0][2] + " (purtari 1). Desaga: nimic. Fara piatra\n")
        for idx in range(1, len(l)):
            fout.write("Pas " + str(idx) + "). ")
            fout.write(l[idx][-1])
            fout.write("Paseste din " + str(l[idx - 1][:2]) + " in " + str(l[idx][:2]) + ". Incaltat " + l[idx][
                2] + " (purtari: " + str(l[idx][3]) + "). ")
            if l[idx][4] is None:
                fout.write("Desaga: nimic. ")
            else:
                fout.write("Desaga " + l[idx][4] + " (purtari " + str(l[idx][5]) + "). ")
            if l[idx][6] == 0:
                fout.write("Fara piatra.\n")
            else:
                fout.write("Cu piatra.\n")

        fout.write("A iesit din pestera. Costul drumului: " + str(cost) + ". Timpul de executie: " + str(
            timp) + ". Noduri generate: " + str(noduri_generate) + ". Numarul maxim de noduri din memorie: " + str(
            nr_max_n) + "\n\n")

        # for inf in l:
        #     print(inf)
        # print(self.id)
        return len(l)

    def contineInDrum(self, infoNodNou):
        nodDrum = self
        while nodDrum is not None:
            if infoNodNou == nodDrum.info:
                return True
            nodDrum = nodDrum.parinte

        return False


class Graph:  # graful problemei
    def __init__(self, start, stone, N, M, colors, objects):
        self.start = start
        self.stone = stone
        self.N = N
        self.M = M
        self.colors = colors
        self.objects = objects

    # va genera succesorii sub forma de noduri in arborele de parcurgere
    def genereazaSuccesori(self, nodCurent):
        listaSuccesori = []
        [posX, posY, shoes_color, uses1, back_shoes_color, uses2, has_stone, _] = nodCurent.info

        for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            if posX+dx < 0 or posX+dx >= self.N or posY+dy < 0 or posY+dy >= self.M:
                continue
            newX = posX + dx
            newY = posY + dy
            new_color = self.colors[newX][newY]
            new_object = self.objects[newX][newY]

            aux = []
            # mai intai verific daca pot merge in noua patratica fara sa mor
            if new_color == shoes_color and uses1 < 3: # continui mai departe folosind cizmele din picioare
                new_state = [newX, newY, shoes_color, uses1+1, back_shoes_color, uses2, has_stone, ""]
                aux.append(new_state)
            if back_shoes_color == new_color: # daca pot schimba cu cei din rucsac
                if shoes_color != back_shoes_color or uses1 == 3: # schimb daca trebuie
                    new_state = [newX, newY, back_shoes_color, uses2+1, shoes_color, uses1, has_stone, "Incalta cizmele din desaga si porneste la drum. "]
                    if uses1 == 3:
                        new_state[4] = None
                        new_state[5] = 0
                        new_state[-1] = "I s-au tocit cizmele. " + new_state[-1] + ". "
                    aux.append(new_state)

            # acum verific ce pot sa obtin din noua patratica
            for new_state in aux:

                if new_object == '@': # iau piatra daca pot
                    new_state[6] = 1
                    if not nodCurent.contineInDrum(new_state):
                        listaSuccesori.append((new_state, self.inadmissibleHeuristic(new_state), nodCurent.g + 1))
                        continue

                if new_object == '*' or new_object == '0': # daca nu e nimic, trec mai departe
                    if not nodCurent.contineInDrum(new_state):
                        listaSuccesori.append((new_state, self.inadmissibleHeuristic(new_state), nodCurent.g + 1))
                        continue

                if back_shoes_color == None: # daca sunt cizme, de orice culoare, si rucsacul e gol, le bag
                    new_state[4] = new_object
                    new_state[5] = 0
                    new_state[-1] = "A gasit cizme " + new_object + ". "
                    if not nodCurent.contineInDrum(new_state):
                        listaSuccesori.append((new_state, self.inadmissibleHeuristic(new_state), nodCurent.g + 1))
                        continue

                if new_state[4] != new_object:
                    if not nodCurent.contineInDrum(new_state):
                        listaSuccesori.append((new_state, self.inadmissibleHeuristic(new_state), nodCurent.g + 1))

                if new_state[4] != new_object or new_state[5] > 0:
                    new_state[4] = new_object
                    new_state[5] = 0
                    new_state[-1] = new_state[-1] + "Schimba cizmele din desaga cu cele din patratel si porneste la drum. "
                    if not nodCurent.contineInDrum(new_state):
                        listaSuccesori.append((new_state, self.inadmissibleHeuristic(new_state), nodCurent.g + 1))

                if new_object == new_color: # daca gasesc cizme care se potrivesc cu culoarea pe care ajung le incalt
                    new_state[2] = new_object
                    new_state[3] = 0
                    if not nodCurent.contineInDrum(new_state):
                        listaSuccesori.append((new_state, self.inadmissibleHeuristic(new_state), nodCurent.g + 1))
                    if new_state[4] != new_state[2]:  # daca are sens sa fac un switch cu ce am in rucsac
                        shoes, ct = new_state[2], new_state[3]
                        new_state[2] = new_object
                        new_state[3] = 0
                        new_state[4] = shoes
                        new_state[5] = ct
                        if new_state[5] == 3: # daca sunt folosite cizmele
                            new_state[4] = None
                            new_state[5] = 0
                            new_state[-1] = "I s-au tocit cizmele. A gasit cizme " + new_object + ". Incalta aceste cizme si porneste la drum. "
                        else:
                            new_state[-1] = new_state[-1] + "Muta cizmele incaltate in desaga si le incalta pe cele din patratel si porneste la drum. "
                        if not nodCurent.contineInDrum(new_state):
                            listaSuccesori.append((new_state, self.inadmissibleHeuristic(new_state), nodCurent.g+1))

        return listaSuccesori

    def banalHeuristic(self, nod_info):
        return abs(nod_info[0] - self.start[0]) + abs(nod_info[1] - self.start[1])

    def manhattanHeuristic(self, nod_info):
        if nod_info[-2] == 1:
            return abs(nod_info[0] - self.start[0]) + abs(nod_info[1] - self.start[1])
        else:
            return (abs(nod_info[0] - self.stone[0]) + abs(nod_info[1] - self.stone[1])) + (
                        abs(nod_info[0] - self.start[0]) + abs(nod_info[1] - self.start[1]))

    def tieBreakingHeuristic(self, nod_info):
        dx1 = nod_info[0] - self.start[0]
        dy1 = nod_info[1] - self.start[1]
        dx2 = self.stone[0] - self.start[0]
        dy2 = self.stone[1] - self.start[1]
        cross = abs(dx1 * dy2 - dx2 * dy1)
        return cross * 0.001

    def inadmissibleHeuristic(self, nod_info):
        if nod_info[-2] == 1:
            return 1000 * self.N * self.M - abs(nod_info[0] - self.start[0]) + abs(nod_info[1] - self.start[1])
        else:
            return 1000 * self.N * self.M - (abs(nod_info[0] - self.stone[0]) + abs(nod_info[1] - self.stone[1])) + (
                    abs(nod_info[0] - self.start[0]) + abs(nod_info[1] - self.start[1]))


def in_list(nod_info, lista):
    for nod in lista:
        if nod_info == nod.info:
            return nod
    return None

def a_star_o(gr, start, timeout, output_file):
    """
    A* optimizat
    :param gr: graful pe care il expandam
    :param start: starea initiala
    :param timeout: dupa cat timp sa se opreasca algoritmul
    :param output_file: unde se vor scrie solutiile
    """
    opened = [NodParcurgere(start, None, 0, 0)]
    closed = []
    continua = True
    start_timer = time.time()
    fout = open(output_file, "w")
    noduri_generate = 0
    nr_max_c = 0

    while len(opened) > 0 and continua:
        if time.time() - start_timer > timeout:
            fout.write("\nTimeout!!!")
            break
        nodCurent = opened.pop(0)
        closed.append(nodCurent)

        if gr.objects[nodCurent.info[0]][nodCurent.info[1]] == '*' and nodCurent.info[-2] == 1:
            nodCurent.afisDrum(nodCurent.g, time.time() - start_timer, noduri_generate, nr_max_c, fout)
            continua = False

        lSuccesori = gr.genereazaSuccesori(nodCurent)
        noduri_generate += len(lSuccesori)

        for nod in lSuccesori:
            nextNod, h, g = nod
            node_open = in_list(nextNod, opened)
            node_parc = NodParcurgere(nextNod, nodCurent, g, g+h)
            if node_open is not None:
                if node_open.f > g + h:
                    opened.remove(node_open)
                    i = 0
                    while i < len(opened) and (opened[i].f < node_parc.f or (opened[i].f == node_parc.f and opened[i].g > node_parc.g)):
                        i += 1
                    opened.insert(i, node_parc)
                continue

            node_closed = in_list(nextNod, closed)
            if node_closed is not None:
                if node_closed.f > g + h:
                    closed.remove(node_closed)
                    i = 0
                    while i < len(opened) and (opened[i].f < node_parc.f or (opened[i].f == node_parc.f and opened[i].g > node_parc.g)):
                        i += 1
                    opened.insert(i, node_parc)
                continue

            i = 0
            while i < len(opened) and (opened[i].f < node_parc.f or (opened[i].f == node_parc.f and opened[i].g > node_parc.g)):
                i += 1
            opened.insert(i, node_parc)

        nr_max_c = max(nr_max_c, len(opened)+len(closed))

    if continua:
        fout.write("No solution!")

    fout.close()
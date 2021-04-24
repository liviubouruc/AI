import time


class NodParcurgere:
    def __init__(self, id, info, cost, parinte):
        self.id = id  # este indicele din vectorul de noduri
        self.info = info  # informatii despre stare
        self.cost = cost
        self.parinte = parinte  # parintele din arborele de parcurgere

    def obtineDrum(self):
        l = [self.info]
        nod = self
        while nod.parinte is not None:
            l.insert(0, nod.parinte.info)
            nod = nod.parinte
        return l

    def afisDrum(self, cost, timp, noduri_generate, nr_max_n, fout):  # returneaza si lungimea drumului
        l = self.obtineDrum()

        fout.write("Pas 0). Incepe drumul cu cizme de culoarea " + l[0][2] + " din locatia " + str(l[0][:2]) + ". Incaltat: " + l[0][2] + " (purtari 1). Desaga: nimic. Fara piatra\n")
        for idx in range(1, len(l)):
            fout.write("Pas " + str(idx) + "). ")
            fout.write(l[idx][-1])
            fout.write("Paseste din " + str(l[idx-1][:2]) + " in " + str(l[idx][:2]) + ". Incaltat " + l[idx][2] + " (purtari: " + str(l[idx][3]) + "). ")
            if l[idx][4] is None:
                fout.write("Desaga: nimic. ")
            else:
                fout.write("Desaga " + l[idx][4] + " (purtari " + str(l[idx][5]) + "). ")
            if l[idx][6] == 0:
                fout.write("Fara piatra.\n")
            else:
                fout.write("Cu piatra.\n")

        fout.write("A iesit din pestera. Costul drumului: " + str(cost) + ". Timpul de executie: " + str(timp) + ". Noduri generate: " + str(noduri_generate) + ". Numarul maxim de noduri din memorie: " + str(nr_max_n) + "\n\n")

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
            # mai intai verific daca pot merge in noua patratica fara sa mor; aceste conditii verifica daca mai are sens sa expandez nodul curent i.e. daca pot avansa
            if new_color == shoes_color and uses1 < 3: # continui mai departe folosind cizmele din picioare
                new_state = [newX, newY, shoes_color, uses1+1, back_shoes_color, uses2, has_stone, ""]
                aux.append(new_state)
            if back_shoes_color == new_color: # daca pot schimba cu cei din rucsac
                if shoes_color != back_shoes_color or uses1 == 3: # schimb daca trebuie
                    new_state = [newX, newY, back_shoes_color, uses2+1, shoes_color, uses1, has_stone, "Incalta cizmele din desaga si porneste la drum. "]
                    if uses1 == 3:
                        new_state[4] = None
                        new_state[5] = 0
                        new_state[-1] = "I s-au tocit cizmele. " + new_state[-1]
                    aux.append(new_state)

            # acum verific ce pot sa obtin din noua patratica
            for new_state in aux:

                if new_object == '@': # iau piatra daca pot
                    new_state[6] = 1
                    if not nodCurent.contineInDrum(new_state):
                        listaSuccesori.append(NodParcurgere(-1, new_state, nodCurent.cost+1, nodCurent))
                        continue

                if new_object == '*' or new_object == '0': # daca nu e nimic, trec mai departe
                    if not nodCurent.contineInDrum(new_state):
                        listaSuccesori.append(NodParcurgere(-1, new_state, nodCurent.cost + 1, nodCurent))
                        continue

                if back_shoes_color == None: # daca sunt cizme, de orice culoare, si rucsacul e gol, le bag
                    new_state[4] = new_object
                    new_state[5] = 0
                    new_state[-1] = "A gasit cizme " + new_object + ". "
                    if not nodCurent.contineInDrum(new_state):
                        listaSuccesori.append(NodParcurgere(-1, new_state, nodCurent.cost + 1, nodCurent))
                        continue
                # daca am ceva in rucsac si sunt si cizme jos
                if new_state[4] != new_object: # pot alege sa continui asa cum sunt acum
                    if not nodCurent.contineInDrum(new_state):
                        listaSuccesori.append(NodParcurgere(-1, new_state, nodCurent.cost + 1, nodCurent))

                if new_state[4] != new_object or new_state[5] > 0: # aleg sa inlocuiesc ce am in ruscac cu ce e jos
                    new_state[4] = new_object
                    new_state[5] = 0
                    new_state[-1] = new_state[-1] + "Schimba cizmele din desaga cu cele din patratel si porneste la drum. "
                    if not nodCurent.contineInDrum(new_state):
                        listaSuccesori.append(NodParcurgere(-1, new_state, nodCurent.cost + 1, nodCurent))

                if new_object == new_color: # daca gasesc cizme care se potrivesc cu culoarea pe care ajung le incalt
                    new_state[2] = new_object
                    new_state[3] = 0
                    if not nodCurent.contineInDrum(new_state):
                        listaSuccesori.append(NodParcurgere(-1, new_state, nodCurent.cost + 1, nodCurent))

                    if new_state[4] != new_state[2]: # daca are sens sa fac un switch cu ce am in rucsac
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
                            listaSuccesori.append(NodParcurgere(-1, new_state, nodCurent.cost + 1, nodCurent))

        return listaSuccesori


def ucs(gr, start, nrSolutiiCautate, timeout, output_file):
    """
    UCS
    :param gr: graful pe care il expandam
    :param start: starea initiala
    :param nrSolutiiCautate: NSOL
    :param timeout: dupa cat timp sa se opreasca algoritmul
    :param output_file: unde se vor scrie solutiile
    """
    c = [NodParcurgere(-1, start, 0, None)]
    continua = True
    start_timer = time.time()
    fout = open(output_file, "w")
    sol_ct = 0
    noduri_generate = 0
    nr_max_c = 0
    found = []

    while len(c) > 0 and continua:
        if time.time() - start_timer > timeout:
            fout.write("\nTimeout!!!")
            break
        nodCurent = c.pop(0)

        if gr.objects[nodCurent.info[0]][nodCurent.info[1]] == '*' and nodCurent.info[-2] == 1:
            drum = nodCurent.obtineDrum()
            if drum in found:
                continue
            fout.write("SOLUTIA " + str(sol_ct+1) + "\n")
            nodCurent.afisDrum(nodCurent.cost, time.time()-start_timer, noduri_generate, nr_max_c, fout)
            found.append(drum)
            sol_ct += 1
            if sol_ct == nrSolutiiCautate:
                continua = False

        lSuccesori = gr.genereazaSuccesori(nodCurent)
        noduri_generate += len(lSuccesori)

        for s in lSuccesori:
            i = 0
            gasit_loc = False
            for i in range(len(c)):
                if c[i].cost >= s.cost:
                    gasit_loc = True
                    break
            if gasit_loc:
                c.insert(i, s)
            else:
                c.append(s)
        nr_max_c = max(nr_max_c, len(c))

    if continua:
        fout.write("No solution!")

    fout.close()

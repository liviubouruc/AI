import sys
import time
import copy
from statistics import median
import pygame

tabla = []

for i in range(3):
    tabla.append(['*' for x in range(5)])
# pozitii initiale caini
tabla[0][1] = 'c'
tabla[1][0] = 'c'
tabla[2][1] = 'c'
# pozitie initiala iepure
tabla[1][4] = 'i'

mutari_invalide = [(0, 0), (2, 0), (0, 4), (2, 4)]  # colturile matricei care nu vor fi afisate
ADANCIME_MAXIMA = 0
nr_mutari_jucator = 0
nr_mutari_calculator = 0
times = []
nodes = []
scor_final_jucator = 0
scor_final_calculator = 0
mutari_totale = 0
t_initial = time.time()


def mutari_valide_caini(linie_curenta, col_curenta, linie_mutare, col_mutare):
    if col_curenta == col_mutare:
        if linie_curenta == linie_mutare - 1 or linie_curenta == linie_mutare + 1:  # muta pe verticala
            return True
    if ((linie_curenta == linie_mutare and col_mutare == col_curenta + 1) or  # muta pe orizontala
            (linie_curenta == linie_mutare + 1 and col_curenta == col_mutare - 1) or  # muta pe diagonala in sus
            (linie_curenta == linie_mutare - 1 and col_curenta == col_mutare - 1)):  # muta pe diagonala in jos
        return True
    return False


def mutari_valide_iepure(linie_curenta, col_curenta, linie_mutare, col_mutare):
    return ((abs(linie_curenta - linie_mutare) +
             abs(col_curenta - col_mutare) == 1) or  # mutare sus-jos, stanga-dreapta
            (linie_curenta == linie_mutare + 1
             and col_curenta == col_mutare - 1) or  # mutare pe diagonala sprea dreapta in sus
            (linie_curenta == linie_mutare - 1
             and col_curenta == col_mutare - 1) or  # mutare pe diagonala spre dreapta in jos
            (linie_curenta == linie_mutare + 1
             and col_curenta == col_mutare + 1) or  # muutare pe diagonala sprea stanga in sus
            (linie_curenta == linie_mutare - 1
             and col_curenta == col_mutare + 1))  # muutare pe diagonala sprea stanga in jos


def distanta_puncte(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def get_pozitii_caini(tabla):
    pozitii_caini = []
    for i in range(len(tabla)):
        for j in range(len(tabla[i])):
            if tabla[i][j] == 'c':
                pozitii_caini.append((i, j))
    return pozitii_caini


def get_pozitie_iepure(tabla_joc):
    for i in range(len(tabla_joc)):
        for j in range(len(tabla_joc[i])):
            if tabla_joc[i][j] == 'i':
                return i, j


def get_castigator(tabla):
    linie_iepure, coloana_iepure = get_pozitie_iepure(tabla)
    pozitii_caini = get_pozitii_caini(tabla)
    linie_caine1, coloana_caine1 = pozitii_caini[0][0], pozitii_caini[0][1]
    linie_caine2, coloana_caine2 = pozitii_caini[1][0], pozitii_caini[1][1]
    linie_caine3, coloana_caine3 = pozitii_caini[2][0], pozitii_caini[2][1]

    # verific daca iepurele a castigat

    # -> daca e in stanga tuturor cainilor
    if ((coloana_iepure < coloana_caine1) and
            (coloana_iepure < coloana_caine2) and
            (coloana_iepure < coloana_caine3)):
        return 'i'
    # -> daca doi caini sunt in dreapta lui si e pe aceeasi coloana cu al treilea
    if ((coloana_iepure == coloana_caine1 and coloana_iepure < coloana_caine2 and
         coloana_iepure < coloana_caine3) or
            (coloana_iepure == coloana_caine2 and coloana_iepure < coloana_caine1 and
             coloana_iepure < coloana_caine3) or
            (coloana_iepure == coloana_caine3 and coloana_iepure < coloana_caine2
             and coloana_iepure < coloana_caine1)):
        return 'i'

    # verific daca cainii au castigat
    if linie_iepure == 1 and coloana_iepure == 4 and coloana_caine1 == coloana_caine2 == coloana_caine3 == 3:
        if ((linie_caine1 == 0 and linie_caine2 == 1 and linie_caine3 == 2) or
                (linie_caine1 == 1 and linie_caine2 == 2 and linie_caine3 == 0) or
                (linie_caine1 == 2 and linie_caine2 == 1 and linie_caine3 == 0)):
            return 'c'
    elif linie_iepure == 0 and coloana_iepure == 2:
        if ((linie_caine1 == 0 and coloana_caine1 == 1 and linie_caine2 == 0
             and coloana_caine2 == 3 and linie_caine3 == 1 and coloana_caine3 == 2) or
                (linie_caine1 == 0 and coloana_caine1 == 3 and linie_caine2 == 1
                 and coloana_caine2 == 2 and linie_caine3 == 0 and coloana_caine3 == 1) or
                (linie_caine1 == 1 and coloana_caine1 == 2 and linie_caine2 == 0
                 and coloana_caine2 == 1 and linie_caine3 == 0 and coloana_caine3 == 3)):
            return 'c'
    elif linie_iepure == 2 and coloana_iepure == 2:
        if ((linie_caine1 == 2 and coloana_caine1 == 1 and linie_caine2 == 1
             and coloana_caine2 == 2 and linie_caine3 == 2 and coloana_caine3 == 3) or
                (linie_caine1 == 1 and coloana_caine1 == 2 and linie_caine2 == 2
                 and coloana_caine2 == 3 and linie_caine3 == 2 and coloana_caine3 == 1) or
                (linie_caine1 == 2 and coloana_caine1 == 3 and linie_caine2 == 2
                 and coloana_caine2 == 1 and linie_caine3 == 1 and coloana_caine3 == 2)):
            return 'c'

    # daca nu a castigat nimeni, returnez False
    return False


def distanta_totala_iepure_caini(tabla):
    lin_iepure, col_iepure = get_pozitie_iepure(tabla)
    pozitii_caini = get_pozitii_caini(tabla)
    lin_caine1, col_caine1 = pozitii_caini[0][0], pozitii_caini[0][1]
    lin_caine2, col_caine2 = pozitii_caini[1][0], pozitii_caini[1][1]
    lin_caine3, col_caine3 = pozitii_caini[2][0], pozitii_caini[2][1]

    distanta_iepure_caine_1 = distanta_puncte(lin_iepure, col_iepure, lin_caine1, col_caine1)
    distanta_iepure_caine_2 = distanta_puncte(lin_iepure, col_iepure, lin_caine2, col_caine2)
    distanta_iepure_caine_3 = distanta_puncte(lin_iepure, col_iepure, lin_caine3, col_caine3)

    return distanta_iepure_caine_1 + distanta_iepure_caine_2 + distanta_iepure_caine_3


class Joc:
    JMIN = None
    JMAX = None
    GOL = '*'

    def __init__(self, tabla_joc=None):
        self.matr = tabla_joc or tabla

    # @classmethod
    # def initializeaza(cls, display, matr, NR_LINII=3, NR_COLOANE=4, dim_celula=100):
    #     cls.display = display
    #     cls.dim_celula = dim_celula
    #     # cls.x_img = pygame.image.load('ics.png')
    #     # cls.x_img = pygame.transform.scale(cls.x_img, (dim_celula, dim_celula))
    #     # cls.zero_img = pygame.image.load('zero.png')
    #     # cls.zero_img = pygame.transform.scale(cls.zero_img, (dim_celula, dim_celula))
    #     # cls.celuleGrid = []  # este lista cu patratelele din grid
    #     for linie in range(NR_LINII):
    #         for coloana in range(NR_COLOANE):
    #             #if (linie+1, coloana) not in mutari_invalide:
    #             #    pygame.draw.aaline(display, (0, 0, 0), (linie*10, coloana*10), ((linie+1)*10, coloana*10), 4)
    #             patr = pygame.Rect(coloana * (dim_celula + 1), linie * (dim_celula + 1), dim_celula, dim_celula)
    #             # cls.celuleGrid.append(patr)
    #             pygame.draw.rect(display, (0, 0, 0), patr, 1)
    #
    # def deseneaza_grid(self, marcaj=None):  # tabla de exemplu este ["#","x","#","0",......]
    #
    #     for ind in range(len(self.matr)):
    #         linie = ind // 3  # // inseamna div
    #         coloana = ind % 3
    #
    #         if marcaj == ind:
    #             # daca am o patratica selectata, o desenez cu rosu
    #             culoare = (255, 0, 0)
    #         else:
    #             # altfel o desenez cu alb
    #             culoare = (255, 255, 255)
    #         pygame.draw.rect(self.__class__.display, culoare, self.__class__.celuleGrid[ind])  # alb = (255,255,255)
    #         if self.matr[ind] == 'x':
    #             self.__class__.display.blit(self.__class__.x_img, (
    #             coloana * (self.__class__.dim_celula + 1), linie * (self.__class__.dim_celula + 1)))
    #         elif self.matr[ind] == '0':
    #             self.__class__.display.blit(self.__class__.zero_img, (
    #             coloana * (self.__class__.dim_celula + 1), linie * (self.__class__.dim_celula + 1)))
    #     pygame.display.flip()  # obligatoriu pentru a actualiza interfata (desenul)
    # # pygame.display.update()

    def final(self):
        return get_castigator(self.matr)

    def mutari(self, jucator_opus):
        l_mutari = []
        for i in range(len(self.matr)):
            for j in range(len(self.matr[i])):
                if self.matr[i][j] == self.GOL and (i, j) not in mutari_invalide:
                    if jucator_opus == 'c':
                        pozitii_caini = get_pozitii_caini(self.matr)
                        lin_caine_1, col_caine_1 = pozitii_caini[0]
                        lin_caine_2, col_caine_2 = pozitii_caini[1]
                        lin_caine_3, col_caine_3 = pozitii_caini[2]
                        if mutari_valide_caini(lin_caine_1, col_caine_1, i, j):
                            self.creare_tabla_noua(lin_caine_1, col_caine_1, i, j, jucator_opus, l_mutari)
                        if mutari_valide_caini(lin_caine_2, col_caine_2, i, j):
                            self.creare_tabla_noua(lin_caine_2, col_caine_2, i, j, jucator_opus, l_mutari)
                        if mutari_valide_caini(lin_caine_3, col_caine_3, i, j):
                            self.creare_tabla_noua(lin_caine_3, col_caine_3, i, j, jucator_opus, l_mutari)
                    else:
                        lin_iepure, col_iepure = get_pozitie_iepure(self.matr)
                        if mutari_valide_iepure(lin_iepure, col_iepure, i, j):
                            self.creare_tabla_noua(lin_iepure, col_iepure, i, j, jucator_opus, l_mutari)
        return l_mutari

    def creare_tabla_noua(self, linie_curenta, col_curenta, linie_mutare, col_mutare, jucator_opus, l_mutari):
        matr_tabla_noua = copy.deepcopy(self.matr)
        matr_tabla_noua[linie_mutare][col_mutare] = jucator_opus
        matr_tabla_noua[linie_curenta][col_curenta] = Joc.GOL
        l_mutari.append(Joc(matr_tabla_noua))

    def __str__(self):
        sir = ''
        for i in range(len(self.matr)):
            for j in range(len(self.matr[i])):
                if (i, j) not in mutari_invalide:
                    sir += self.matr[i][j] + '\t'
                else:
                    sir += ' \t'
            sir += '\n'
        sir += "\n" + ('-' * 30) + "\n"
        return sir


class Stare:
    """
    Clasa folosita de algoritmii minimax si alpha-beta
    Are ca proprietate tabla de joc
    Functioneaza cu conditia ca in cadrul clasei Joc sa fie definiti JMIN si JMAX (cei doi jucatori posibili)
    De asemenea cere ca in clasa Joc sa fie definita si o metoda numita mutari() care ofera lista cu configuratiile posibile in urma mutarii unui jucator
    """

    def __init__(self, tabla_joc, j_curent, adancime, parinte=None, scor=0):
        self.tabla_joc = tabla_joc
        self.j_curent = j_curent
        # adancimea in arborele de stari
        self.adancime = adancime
        # scorul starii (daca e finala) sau al celei mai bune stari-fiice (pentru jucatorul curent)
        self.scor = scor
        # lista de mutari posibile din starea curenta
        self.mutari_posibile = []
        # cea mai buna mutare din lista de mutari posibile pentru jucatorul curent
        self.stare_aleasa = None

    def estimeaza_scor1(self, adancime):
        t_final = Joc.final(self.tabla_joc)
        if t_final == Joc.JMAX:
            return 99 + adancime
        elif t_final == Joc.JMIN:
            return -99 - adancime
        else:
            return -distanta_totala_iepure_caini(self.tabla_joc.matr)

    def estimeaza_scor2(self, adancime):
        t_final = Joc.final(self.tabla_joc)
        if t_final == Joc.JMAX:
            return 99 + adancime
        elif t_final == Joc.JMIN:
            return -99 - adancime
        else:
            pozitii_caini = get_pozitii_caini(tabla)
            cols = sorted([pozitii_caini[0][1], pozitii_caini[1][1], pozitii_caini[2][1]])
            return (cols[0]+2)*10+(cols[1]+2)+cols[2] + distanta_totala_iepure_caini(self.tabla_joc.matr)

    def jucator_opus(self):
        if self.j_curent == Joc.JMIN:
            return Joc.JMAX
        else:
            return Joc.JMIN

    def mutari(self):
        l_mutari = self.tabla_joc.mutari(self.j_curent)
        juc_opus = self.jucator_opus()
        l_stari_mutari = [Stare(mutare, juc_opus, self.adancime - 1, parinte=self) for mutare in l_mutari]
        return l_stari_mutari

    def __str__(self):
        sir = str(self.tabla_joc) + "(Jucatorul curent:" + self.j_curent + ")\n"
        return sir


""" Algoritmul MinMax """
def min_max(stare):
    if stare.adancime == 0 or stare.tabla_joc.final() or mutari_totale >= 10:
        stare.scor = stare.estimeaza_scor1(stare.adancime)
        return stare

    # calculez toate mutarile posibile din starea curenta
    stare.mutari_posibile = stare.mutari()

    # aplic algoritmul minimax pe toate mutarile posibile (calculand astfel subarborii lor)
    mutari_scor = [min_max(mutare) for mutare in stare.mutari_posibile]

    if stare.j_curent == Joc.JMAX:
        # daca jucatorul e JMAX aleg starea-fiica cu scorul maxim
        stare.stare_aleasa = max(mutari_scor, key=lambda x: x.scor)
    else:
        # daca jucatorul e JMIN aleg starea-fiica cu scorul minim
        stare.stare_aleasa = min(mutari_scor, key=lambda x: x.scor)
    stare.scor = stare.stare_aleasa.scor
    return stare


def alpha_beta(alpha, beta, stare):
    if stare.adancime == 0 or stare.tabla_joc.final() or mutari_totale >= 10:
        stare.scor = stare.estimeaza_scor2(stare.adancime)
        return stare

    if alpha > beta:
        return stare  # este intr-un interval invalid deci nu o mai procesez

    stare.mutari_posibile = stare.mutari()
    stare.mutari_posibile = sorted(stare.mutari_posibile, key=lambda x: x.scor)

    if stare.j_curent == Joc.JMAX:
        scor_curent = float('-inf')

        for mutare in stare.mutari_posibile:
            # calculeaza scorul
            stare_noua = alpha_beta(alpha, beta, mutare)

            if scor_curent < stare_noua.scor:
                stare.stare_aleasa = stare_noua
                scor_curent = stare_noua.scor
            if alpha < stare_noua.scor:
                alpha = stare_noua.scor
                if alpha >= beta:
                    break

    elif stare.j_curent == Joc.JMIN:
        scor_curent = float('inf')

        for mutare in stare.mutari_posibile:

            stare_noua = alpha_beta(alpha, beta, mutare)

            if scor_curent > stare_noua.scor:
                stare.stare_aleasa = stare_noua
                scor_curent = stare_noua.scor

            if beta > stare_noua.scor:
                beta = stare_noua.scor
                if alpha >= beta:
                    break
    stare.scor = stare.stare_aleasa.scor
    return stare


def afis_daca_final(stare_curenta):
    if mutari_totale >= 10:
        print("A castigat i!")
        print("Scor jucator: " + str(scor_final_jucator))
        print("Scor calculator: " + str(scor_final_calculator))
        print("Numar mutari jucator: " + str(nr_mutari_jucator))
        print("Numar mutari calculator: " + str(nr_mutari_calculator))
        print("Timpul minim de gandire a calculatorului (ms): " + str(min(times)))
        print("Timpul maxim de gandire a calculatorului (ms): " + str(max(times)))
        print("Timpul mediu de gandire a calculatorului (ms): " + str(sum(times)/nr_mutari_calculator))
        print("Mediana timpurilor de gandire a calculatorului (ms): " + str(median(times)))
        print("Numarul minim de noduri generate: " + str(min(nodes)))
        print("Numarul maxim de noduri generate: " + str(max(nodes)))
        print("Numarul mediu de noduri generate: " + str(sum(nodes) / nr_mutari_calculator))
        print("Mediana numerelor noduri generate: " + str(median(nodes)))
        return True

    final = stare_curenta.tabla_joc.final()
    if final:
        if final == "remiza":
            print("Remiza!")
        else:
            print("A castigat " + final)
            print("Scor jucator: " + str(scor_final_jucator))
            print("Scor calculator: " + str(scor_final_calculator))
            print("Numar mutari jucator: " + str(nr_mutari_jucator))
            print("Numar mutari calculator: " + str(nr_mutari_calculator))
            print("Timpul minim de gandire a calculatorului (ms): " + str(min(times)))
            print("Timpul maxim de gandire a calculatorului (ms): " + str(max(times)))
            print("Timpul mediu de gandire a calculatorului (ms): " + str(sum(times) / nr_mutari_calculator))
            print("Mediana timpurilor de gandire a calculatorului (ms): " + str(median(times)))
            print("Numarul minim de noduri generate: " + str(min(nodes)))
            print("Numarul maxim de noduri generate: " + str(max(nodes)))
            print("Numarul mediu de noduri generate: " + str(sum(nodes) / nr_mutari_calculator))
            print("Mediana numerelor noduri generate: " + str(median(nodes)))
        return True

    return False


def get_pozitie_caine_de_mutat(stare_curenta):
    raspuns_valid = False
    linia_caine, coloana_caine = None, None
    while not raspuns_valid:
        # ia pozita cainelui din clic
        try:
            linia_caine = int(input("Linie caine de mutat= "))
            coloana_caine = int(input("Coloana caine de mutat = "))

            if linia_caine in range(0, 3) and coloana_caine in range(0, 5) \
                    and (linia_caine, coloana_caine) not in mutari_invalide:
                if stare_curenta.tabla_joc.matr[linia_caine][coloana_caine] != 'c':
                    raspuns_valid = False
                    print("Nu exista caine in pozitia (" + str(linia_caine) + ", " + str(
                        coloana_caine) + ").")
                else:
                    raspuns_valid = True
            else:
                print("Linie sau coloana invalida (trebuie sa fie unul dintre numerele 0, 1, 2).")
        except ValueError:
            print("Linia si coloana trebuie sa fie numere intregi")
    return linia_caine, coloana_caine


def get_pozitia_urmatoare(stare_curenta):
    raspuns_valid = False
    linia_urmatoare, coloana_urmatoare = None, None
    while not raspuns_valid:
        # ia pozitia unde s-a dat clic
        try:
            linia_urmatoare = int(input("Linia urmatoare = "))
            coloana_urmatoare = int(input("Coloana urmatoare = "))

            if linia_urmatoare in range(0, 3) and coloana_urmatoare in range(0, 5) and \
                    (linia_urmatoare, coloana_urmatoare) not in mutari_invalide:
                if stare_curenta.tabla_joc.matr[linia_urmatoare][coloana_urmatoare] == Joc.GOL:
                    raspuns_valid = True
                else:
                    print("Exista deja un simbol in pozitia ceruta.")
            else:
                print("Linie sau coloana invalida (trebuie sa fie unul dintre numerele 0,1,2).")

        except ValueError:
            print("Linia si coloana trebuie sa fie numere intregi")
    return linia_urmatoare, coloana_urmatoare


def main():
    global mutari_totale
    global nr_mutari_jucator
    global nr_mutari_calculator
    global times
    global nodes
    global scor_final_jucator
    global scor_final_calculator
    global ADANCIME_MAXIMA
    tip_algoritm = 0
    # initializare algoritm
    raspuns_valid = False
    while not raspuns_valid:
        tip_algoritm = input("Algorimul folosit? (raspundeti cu 1 sau 2)\n 1.Minimax\n 2.Alpha-beta\n ")
        if tip_algoritm in ['1', '2']:
            raspuns_valid = True
        else:
            print("Nu ati ales o varianta corecta.")

    # initializare jucatori
    raspuns_valid = False
    while not raspuns_valid:
        Joc.JMIN = input("Doriti sa jucati cu c sau cu i? \n").lower()
        if Joc.JMIN in ['c', 'i']:
            raspuns_valid = True
        else:
            print("Raspunsul trebuie sa fie c sau i.")
    Joc.JMAX = 'i' if Joc.JMIN == 'c' else 'c'

    # alegere dificultate
    dificultate = ''
    while dificultate not in ['incepator', 'mediu', 'avansat']:
        dificultate = input("Alegeti dificultatea: incepator, mediu sau avansat.\n")

    if dificultate == 'incepator':
        ADANCIME_MAXIMA = 4
    elif dificultate == 'mediu':
        ADANCIME_MAXIMA = 6
    elif dificultate == 'avansat':
        ADANCIME_MAXIMA = 8

    # initializare tabla
    tabla_curenta = Joc()
    print("Tabla initiala")
    print(str(tabla_curenta))
    # creare stare initiala
    stare_curenta = Stare(tabla_curenta, 'c', ADANCIME_MAXIMA)

    # pygame.init()
    # pygame.display.set_caption("Bouruc Petru-Liviu - Hares and Hound")
    # ecran = pygame.display.set_mode(size=(302, 302))
    # Joc.initializeaza(ecran, tabla_curenta)

    while True:
        if stare_curenta.j_curent == Joc.JMIN:
            renunt = ''
            while renunt not in ['da', 'nu']:
                renunt = input("Doriti sa iesiti? (da/nu)")
            if renunt == 'da':
                print("Scor jucator: " + str(scor_final_jucator))
                print("Scor calculator: " + str(scor_final_calculator))
                print("Numar mutari jucator: " + str(nr_mutari_jucator))
                print("Numar mutari calculator: " + str(nr_mutari_calculator))
                print("Timpul minim de gandire a calculatorului (ms): " + str(min(times)))
                print("Timpul maxim de gandire a calculatorului (ms): " + str(max(times)))
                print("Timpul mediu de gandire a calculatorului (ms): " + str(sum(times) / nr_mutari_calculator))
                print("Mediana timpurilor de gandire a calculatorului (ms): " + str(median(times)))
                print("Numarul minim de noduri generate: " + str(min(nodes)))
                print("Numarul maxim de noduri generate: " + str(max(nodes)))
                print("Numarul mediu de noduri generate: " + str(sum(nodes) / nr_mutari_calculator))
                print("Mediana numerelor noduri generate: " + str(median(nodes)))
                return

            # muta jucatorul
            if Joc.JMIN == 'c':
                linia_caine_de_mutat, coloana_caine_de_mutat = get_pozitie_caine_de_mutat(stare_curenta)
                linie_urmatoare_caine, coloana_urmatoare_caine = get_pozitia_urmatoare(stare_curenta)

                while not mutari_valide_caini(linia_caine_de_mutat, coloana_caine_de_mutat,
                                              linie_urmatoare_caine, coloana_urmatoare_caine):
                    print("Catelusii se pot muta pe verticala (in sus sau in jos)," +
                          " pe orizontala (doar inainte), pe diagonala (doar de la stanga la dreapta).")
                    linie_urmatoare_caine, coloana_urmatoare_caine = get_pozitia_urmatoare(stare_curenta)

                stare_curenta.tabla_joc.matr[linie_urmatoare_caine][coloana_urmatoare_caine] = Joc.JMIN
                stare_curenta.tabla_joc.matr[linia_caine_de_mutat][coloana_caine_de_mutat] = Joc.GOL
            elif Joc.JMIN == 'i':
                lin_actuala_iepure, col_actuala_iepure = get_pozitie_iepure(stare_curenta.tabla_joc.matr)
                lin_urmatoare_iepure, col_urmatoare_iepure = get_pozitia_urmatoare(stare_curenta)

                while not mutari_valide_iepure(lin_actuala_iepure, col_actuala_iepure,
                                               lin_urmatoare_iepure, col_urmatoare_iepure):
                    print("Iepurele se poate muta doar o pozitie")
                    lin_urmatoare_iepure, col_urmatoare_iepure = get_pozitia_urmatoare(stare_curenta)

                stare_curenta.tabla_joc.matr[lin_urmatoare_iepure][col_urmatoare_iepure] = Joc.JMIN
                stare_curenta.tabla_joc.matr[lin_actuala_iepure][col_actuala_iepure] = Joc.GOL

            # afisarea starii jocului in urma mutarii utilizatorului
            print("\nTabla dupa mutarea jucatorului")
            print(str(stare_curenta))

            scor_final_jucator = stare_curenta.scor
            nr_mutari_jucator += 1
            mutari_totale += 1

            # testez daca jocul a ajuns intr-o stare finala
            # si afisez un mesaj corespunzator in caz ca da
            if afis_daca_final(stare_curenta):
                break

            # S-a realizat o mutare. Schimb jucatorul cu cel opus
            stare_curenta.j_curent = stare_curenta.jucator_opus()

        # --------------------------------
        else:  # jucatorul e JMAX (calculatorul)
            # Mutare calculator
            # preiau timpul in milisecunde de dinainte de mutare
            t_inainte = int(round(time.time() * 1000))

            if tip_algoritm == '1':
                stare_actualizata = min_max(stare_curenta)
                print("Estimare scor aleasa: " + str(stare_actualizata.scor))
                print("Noduri generate: " + str(len(stare_actualizata.mutari_posibile)))
                nodes.append(len(stare_actualizata.mutari_posibile))
            else:  # tip_algoritm==2
                stare_actualizata = alpha_beta(-500, 500, stare_curenta)
                print("Estimare scor aleasa: " + str(stare_actualizata.scor))
                print("Noduri generate: " + str(len(stare_actualizata.mutari_posibile)))
                nodes.append(len(stare_actualizata.mutari_posibile))

            stare_curenta.tabla_joc = stare_actualizata.stare_aleasa.tabla_joc
            print("Tabla dupa mutarea calculatorului")
            print(str(stare_curenta))

            # preiau timpul in milisecunde de dupa mutare
            t_dupa = int(round(time.time() * 1000))
            times.append(t_dupa-t_inainte)
            print("Calculatorul a \"gandit\" timp de " + str(t_dupa - t_inainte) + " milisecunde.")
            if afis_daca_final(stare_curenta):
                break

            # S-a realizat o mutare. Schimb jucatorul cu cel opus
            stare_curenta.j_curent = stare_curenta.jucator_opus()
            scor_final_calculator = stare_curenta.scor
            nr_mutari_calculator += 1
            mutari_totale += 1

        print("Mutari totale: " + str(mutari_totale) + '\n')


if __name__ == "__main__":
    main()
    t_final = time.time()
    ms = round(1000 * (t_final - t_initial))
    print("Timp total de rulare: " + str(t_final - t_initial))
    # while True:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #
    #             pygame.quit()
    #             sys.exit()



import os
import ucs
import a_star
import a_star_o
import ida


def main():
    input_folder = input("Enter the folder with the input files: ")
    if os.path.exists(input_folder) == 0:
        print("Input folder does not exist!")
        return
    files = [file for file in os.listdir(input_folder)]

    output_folder = input("Enter the folder with the output files: ")
    if os.path.exists(output_folder) == 0:
        os.mkdir(output_folder)

    NSOL = int(input("Enter the number of solutions: "))
    if NSOL <= 0:
        print("Can't be negative or 0!")
        return
    timeout = float(input("Enter the timout: "))
    if timeout <= 0:
        print("Can't be negative or 0!")
        return

    print("Select the algorithm")
    print("1.UCS")
    print("2.A*")
    print("3.A* optimised")
    print("4.IDA*")
    alg = int(input())
    if not 1 <= alg <= 4:
        print("Wrong choice!")
        return

    for file in files:
        with open(input_folder + "/" + file) as fin:
            lines = [line.split() for line in fin.readlines()]
            N = (len(lines)-1) // 2
            M = len(lines[0])

            for line in lines:
                if len(line) != M and line != []:
                    print("Wrong input matrix! Different line lengths!")
                    return

        colors = lines[:N]
        objects = lines[N+1:]
        start = None
        stone = None

        for i in range(N):
            for j in range(M):
                if objects[i][j] == '*':
                    start = (i, j)
                if objects[i][j] == '@':
                    stone = (i, j)
        if start is None or stone is None:
            print("No start/stone!")
            continue

        if alg == 1:
            gr = ucs.Graph(start, stone, N, M, colors, objects)
            ucs.ucs(gr, [start[0], start[1], colors[start[0]][start[1]], 1, None, 0, 0, ""], NSOL, timeout, output_folder+"/output_"+file)
        elif alg == 2:
            gr = a_star.Graph(start, stone, N, M, colors, objects)
            a_star.a_star(gr, [start[0], start[1], colors[start[0]][start[1]], 1, None, 0, 0, ""], NSOL, timeout, output_folder+"/output_"+file)
        elif alg == 3:
            gr = a_star_o.Graph(start, stone, N, M, colors, objects)
            a_star_o.a_star_o(gr, [start[0], start[1], colors[start[0]][start[1]], 1, None, 0, 0, ""], timeout, output_folder+"/output_"+file)
        else:
            gr = a_star_o.Graph(start, stone, N, M, colors, objects)
            ida.ida(gr, [start[0], start[1], colors[start[0]][start[1]], 1, None, 0, 0, ""], timeout, output_folder + "/output_" + file)


if __name__ == "__main__":
    main()

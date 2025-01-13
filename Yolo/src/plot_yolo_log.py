import argparse 
import sys
import matplotlib.pyplot as plt

def main(argv):

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "log_file",
        help = "path to log file"
        )


    numbers = {'1','2','3','4','5','6','7','8','9'}

    iters = []
    loss = []
    lr = []

    fig,ax = plt.subplots()

    iters = []
    loss = []
    lr = []

    args = parser.parse_args()

    f = open(args.log_file)

    lines = [line.rstrip("\n") for line in f.readlines()]
    prev_line = ""
    # for line in lines:
    #     args = line.split(' ')
    #     # print(len(args))
    #     if len(args) > 2:
    #         print(args[1].split(':')[0])
    #     if args[0][-1:]==':' and args[0][0] in numbers :
    #         # print(args)
    #         if args[0][:-1] =='5R' or args[2] == 'weights'or args[0][:-1] =='75R':
    #             continue
    #         lr.append(args[4])
    #         iters.append(int(args[0][:-1]))
    #         loss.append(float(args[2]))
    for line in lines:
        if 'Last accuracy' in line:
            print('line',line)
        args = line.split(' ')
        try:
            if args[0] == '' and args[1][-1:] == ':':
                #print('arg', args)
                iters.append(int(args[1][:-1]))
                loss.append(float(args[3]))
        except Exception as E:
            #print(E)
            continue

    iters = iters[1000:]
    loss = loss[1000:]
    # print(iters[0])
    # lr = lr[1000:]
    # print(iters)
    # print(loss)
    # print('lens', len(iters), len(loss))
    ax.plot(iters, loss)

    plt.xlabel('iters')
    plt.ylabel('loss')
    plt.grid()

    ticks = range(0,250,10)
    plt.savefig('veh_detectionLP_yv4.png')
    plt.show()
    # ax.set_yticks(ticks)
    plt.show()
    
if __name__ == "__main__":
    main(sys.argv)

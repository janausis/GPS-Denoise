from pykalman import KalmanFilter
import numpy as np
import time
import csv
from datetime import datetime
from geopy import distance
import pandas as pd
from trianglesolver import solve
from math import degrees
from argparse import ArgumentParser, ArgumentTypeError
import os

# System call
os.system("")

# Class of different styles
class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


args = ArgumentParser()
args.add_argument("input",nargs='?', default="None", help="input File (Location History.json)")
args.add_argument("output",nargs='?', default="None", help="Output File (will be overwritten!)")
args.add_argument("-d", "--degree", default=30, type=int, help="The degree in which a line has to be to be recognised as a GPS spike (must be above 10 and below 350)")
args.add_argument("-s", "--smooth", default=30, type=int, help="Amount of point which are smoothed simultaniously (Higher takes longer, to low is uneffective, must be above 10)")
args = args.parse_args()


FileInput = args.input
if FileInput == "None":
    from tkinter import filedialog
    from tkinter import *

    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir = os.getcwd(),title = "Input File",filetypes = (("csv","*.csv"),("all files","*.*")))
    FileInput = root.filename
    root.destroy()

    if list(FileInput) == []:
        print(style.RED + "No input file!" + style.RESET)
        input("\n< " + style.UNDERLINE + "press enter to exit" + style.RESET + " > ")
        exit()

    print(style.YELLOW + f"Input File:" + style.WHITE + f" {FileInput}" + style.RESET)


output = args.output
if output == "None":
    from tkinter import filedialog
    from tkinter import *

    root = Tk()
    root.filename = filedialog.asksaveasfile(mode='w', defaultextension=".csv", initialdir = os.getcwd(),title = "Output File",filetypes = (("csv","*.csv"),("all files","*.*")))
    try:
        output = root.filename.name
        root.destroy()

        try:
            root.destroy()
        except:
            pass


        if output == None:
            output = f"{os.getcwd()}\\out.csv"
            print(style.YELLOW + f"Output File: " + style.RED + "No output file!" + style.YELLOW + f" Defaulting to " + style.RESET + f"{output}\n\n")

        else:
            print(style.YELLOW + f"Output File:" + style.WHITE + f" {output}\n\n" + style.RESET)

    except:
        try:
            root.destroy()
        except:
            pass
        output = f"{os.getcwd()}\\out.csv"
        print(style.YELLOW + f"Output File: " + style.RED + "No output file!" + style.YELLOW + f" Defaulting to " + style.RESET + f"{output}\n\n")




degTresh = args.degree
if degTresh < 10:
    degTresh = 10
if degTresh > 350:
    degTresh = 350
degTresh = float(degTresh)

smooth = args.smooth
if smooth < 10:
    smooth = 10
smooth = float(smooth / 2)

added = 0
all = 0
cleared = 0
bad = 0
denoise = [""]
global total
total = 0


count = csv.DictReader(open(FileInput, "r+"), fieldnames=["Time", "Latitude", "Longitude"])
total = sum(1 for row in count)
if total < 1:
    print(style.RED + f"CSV {FileInput} is empty!" + style.RESET)
    input("\n< " + style.UNDERLINE + "press enter to exit" + style.RESET + " > ")
    exit()



def printerLog(cleared, added, all, bad):
    global total
    percent = round((all/total)*100)
    print(style.YELLOW + "\rlocations cleared: " + style.RESET + str(cleared) + style.GREEN + "   locations written: " + style.RESET + str(added) + "   " + style.RESET + style.BLUE + "locations read:"  + style.RESET + " " + str(all) + style.RESET + style.RED + "   locations filtered: "  + style.RESET + str(bad) + f"   | {str(percent)}% ", end="")


print(style.YELLOW + "Progress:" + style.RESET)
printerLog(cleared, added, all, bad)


pre_line = ""
pre_line2 = ""
f = csv.DictReader(open(FileInput, "r+"), fieldnames=["Time", "Latitude", "Longitude"])
next(f, None)


clean = csv.DictWriter(open(output, "w", newline=''), fieldnames = ["Time", "Latitude", "Longitude"])
clean.writerow({"Time": "Time", "Latitude": "Latitude", "Longitude": "Longitude"})
stop = 0

def DenoisePath(denoise):
    global all
    global added
    global cleared
    global bad
    global stop
    last = denoise[-1]

    finalArray = []
    for i in denoise:
        if i != "":
            i = dict(i)
            measureArray = np.asarray((float(i["Latitude"]), float(i["Longitude"])))
            finalArray.append(measureArray)


    measurements = np.asarray(finalArray)



    initial_state_mean = [measurements[0, 0],
                          0,
                          measurements[0, 1],
                          0]

    transition_matrix = [[1, 1, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]]

    observation_matrix = [[1, 0, 0, 0],
                          [0, 0, 1, 0]]



    kf1 = KalmanFilter(transition_matrices = transition_matrix,
                      observation_matrices = observation_matrix,
                      initial_state_mean = initial_state_mean)

    kf1 = kf1.em(measurements, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)


    kf2 = KalmanFilter(transition_matrices = transition_matrix,
                      observation_matrices = observation_matrix,
                      initial_state_mean = initial_state_mean,
                      observation_covariance = 10*kf1.observation_covariance,
                      em_vars=['transition_covariance', 'initial_state_covariance'])

    kf2 = kf2.em(measurements, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances)  = kf2.smooth(measurements)


    for i in range(0, len(smoothed_state_means[:, 0])):
        if denoise[i] != "" and denoise[i] != last:
            k = smoothed_state_means[:, 0][i]
            j = smoothed_state_means[:, 2][i]
            l = denoise[i]
            added = added + 1
            clean.writerow({"Time": dict(l)["Time"], "Latitude": k, "Longitude": j})
        elif denoise[i] == "":
            added = added + 1
        #clean.writerow({"Time": cur_line["Time"], "Latitude": smoothed_state_means[:, 0], "Longitude": smoothed_state_means[:, 2]})


    #print(f"DENOISED {len(denoise)}")
    printerLog(cleared, added, all, bad)
    if cleared != added:
        stop = stop + 1
        if stop == 3:
            #print("OUT OF SYNC")
            pass

    return denoise[-1]



def Clearup(cur_line, pre_line, pre_line2):
    global all
    global added
    global cleared
    global bad
    all = all + 1
    #print(cur_line, pre_line, pre_line2)

    coords_1 = (float(cur_line["Latitude"]), float(cur_line["Longitude"]))
    coords_2 = (float(pre_line["Latitude"]), float(pre_line["Longitude"]))
    coords_3 = (float(pre_line2["Latitude"]), float(pre_line2["Longitude"]))

    d1 = distance.distance(coords_1, coords_2)
    d2 = distance.distance(coords_2, coords_3)
    d3 = distance.distance(coords_3, coords_1)
    d1 = float(str(d1)[:-3])*1000
    d2 = float(str(d2)[:-3])*1000
    d3 = float(str(d3)[:-3])*1000
    if d1 == 0.0:
        d1 = 0.01
    if d2 == 0.0:
        d2 = 0.01
    if d3 == 0.0:
        d3 = 0.01

    ### Filter by Velocity ###
    FMT = "%Y-%m-%d %H:%M:%S"
    Time = (datetime.strptime(cur_line["Time"], FMT) - datetime.strptime(pre_line["Time"], FMT)).total_seconds()
    if Time == 0:
        Time = 0.01
    Velocity = d1 / Time


    ### Filter by angle to last and next Point ###
    if d1 >= 0.01 and d2 >= 0.01 and d3 >= 0.01:
        try:
            a,b,c,A,B,C = solve(a=d1, b=d2, c=d3)
            A,B,C = degrees(A), degrees(B), degrees(C)
            if (360.0 - degTresh) < C or C < degTresh:
                spike= True
            else:
                spike = False
        except:
            with open("log.txt", "a") as myfile:
                myfile.write("\n\n")
                myfile.write(str(all))
                myfile.write("\n")
                myfile.write(str(d1))
                myfile.write(str(d2))
                myfile.write(str(d3))
            spike = False
    else:
        spike = False

    if spike == True:
        #print(f"SPIKE  {len(denoise)}")
        bad = bad + 1
        return "prev"

    elif Velocity < 800:
        #print(f"OK  {len(denoise)}")
        cleared = cleared + 1
        return "append"

    else:
        #print(f"ELSE  {len(denoise)}")
        bad = bad + 1
        return "leave"



for cur_line in f:
    if pre_line == "" or pre_line2 == "":
        pre_line = cur_line
        pre_line2 = pre_line
        clean.writerow({"Time": cur_line["Time"], "Latitude": cur_line["Latitude"], "Longitude": cur_line["Longitude"]})
        continue

    clearup = Clearup(cur_line, pre_line, pre_line2)
    printerLog(cleared, added, all, bad)

    if clearup == "append":
        denoise.append({"Time": cur_line["Time"], "Latitude": cur_line["Latitude"], "Longitude": cur_line["Longitude"]})
        if len(denoise) > smooth -1:
            last = DenoisePath(denoise)
            denoise = [last]

    elif clearup == "prev":
        denoise = denoise[:-1]
        denoise.append({"Time": cur_line["Time"], "Latitude": cur_line["Latitude"], "Longitude": cur_line["Longitude"]})
        if len(denoise) > smooth -1:
            last = DenoisePath(denoise)
            denoise = [last]

    pre_line2 = pre_line
    pre_line = cur_line



try:
    DenoisePath(denoise)
except:
    pass
denoise = []
printerLog(cleared, added, all, bad)
print("")
print(style.GREEN + f"\n\nDone! " + style.WHITE + f"{round((bad/all)*100)}%" + style.GREEN + " of points have been filtered out." + style.RESET)
input("\n< " + style.UNDERLINE + "press enter to exit" + style.RESET + " > ")

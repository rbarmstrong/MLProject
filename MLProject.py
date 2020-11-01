def readInputFile(fileName):
    outputList = [];
    file = open(fileName, "r")
    for line in file:
        outputList.append(line[:len(line)-1])#the -1 gets rid of newline characters
    return outputList




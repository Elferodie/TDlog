import random

Objects = ["book","emerald", "gold", "key", "map", "owl", "ring", "sword"]

def generateDynamicTiles():
    dynamicList = []

    for tile in range(16):
        newTile = Tile(None,None,False,[1,1,0,0],"l",None)
        dynamicList.append(newTile)

    for tile in range(12):
        newTile = Tile(None,None,False,[1,0,1,0],"i",None)
        dynamicList.append(newTile)

    for tile in range(6):
        newTile = Tile(None,None,False,[1,1,1,0],"t",None)
        dynamicList.append(newTile)

    return dynamicList

def placeStaticTiles():

def placeDynamicTiles(dynamicList):
    for i in range(7):
        for j in range(7):
            if (i%2 == 1 or j%2 == 1):
                index = random.randint(0,len(dynamicList)-1)
                self._tilesMatrix[i][j] = dynamicList[index]
                dynamicList.pop(index)

    return dynamicList #only the outtile is left in dynamicList

def displayBoard():
    for i in range(7):
        for j in range(7):
            displayTile(self._Matrix[i][j])

def displayTile():








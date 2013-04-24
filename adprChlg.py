import math, random
from types import*

import visualize
import Grid
import Queue

Grid = Grid.Grid


def simulation(startAnts, worldSize, perBlocked, perFood, printActions=False):
    """
    main routine that runs a simulation of an ant world
    worldSize represents the dimensions of the world, perBlocked
    represents the percent of tiles blocked, and perFood represents
    the percentage of tiles seeded with food

    startAnts: an integer representing the starting number of ants
    worldSize: an integer representing the size of the world
    perBlocked: a floating point number representing the percentage of un-travelable tiles
    perFood: a floating point number representing the percentage of tiles seeded with food
    printActions: a boolean controlling whether to print ant actions to output
    """

    assert type(startAnts) is IntType 

    dirs = dict([("NORTH",(-1,0)),("EAST",(0,1)),("SOUTH",(1,0)),("WEST",(0,-1))])

    world = worldMap(worldSize, worldSize, perBlocked, perFood)

    anim = visualize.antVisualization(world)
        
    currSurroundings = antSurroundings(world.getSurroundings(0,0))

    ## spawn ant at ant mound
    for row in xrange(world.getWorldHeight()):
        for col in xrange(world.getWorldWidth()):
            if world.getTile(row, col).isAntHome():
                for i in xrange(startAnts):
                    world.getTile(row, col).addAnt(Ant(printActions))
               
    iters = 0

    while(iters < 3500):
        ## hash map used to track ants
        worldCopy = {}
        for row in xrange(world.getWorldHeight()):
            for col in xrange(world.getWorldWidth()):
                ## spawn a new ant every 10th turn
                if world.getTile(row, col).isAntHome() and iters%10 == 0:
                    world.getTile(row, col).addAnt(Ant(printActions))
                
                if world.getTile(row, col).getAnts() != []:
                    currSurroundings.updatePosition(world.getSurroundings(row, col))

                    for ant in tuple(world.getTile(row, col).getAnts()):

                        action = ant.getAction(currSurroundings)

                        if action == "HALT":
                            pass
                            ## track the ant's position
                            if (row, col) not in worldCopy:
                                worldCopy[(row, col)] = []
                            worldCopy[(row, col)].append(ant)
                        elif action == "GATHER":
                            assert ant.getAntHasFood() == False
                            ## if the ant does not have food and food is present on the tile
                            if not ant.getAntHasFood() and world.getTile(row, col).getAmountOfFood() > 0:
                                ## decrement the amount of food at the tile
                                world.getTile(row, col).takeFood()
                                ## set the ant's hasFood field to true
                                ant.antGather()
                                ## track the ant's position
                                if (row, col) not in worldCopy:
                                    worldCopy[(row, col)] = []
                                worldCopy[(row, col)].append(ant)
                        elif action == "DROPOFF":
                            if ant.getAntHasFood() and world.getTile(row, col).isAntHome():
                                ## set the ant's hasFood field back to false
                                ant.antDropoff()
                                ## increment the food at the home tile
                                world.getTile(row, col).addFood()
                                ## track the ants position 
                                if (row, col) not in worldCopy:
                                    worldCopy[(row, col)] = []
                                worldCopy[(row, col)].append(ant)
                        else: 
                            newRow = row + (dirs[action])[0]
                            newCol = col + (dirs[action])[1]
                            ## validate the new row, col position is in the world
                            ## ants should never return a movement that would be off the map  
                            assert newRow >= 0 and newRow < world.getWorldHeight() and newCol >= 0 and newCol < world.getWorldWidth()
                            ## validate the new row, col position is not blocked
                            assert world.getTile(newRow, newCol).isTravelable() 
                            if world.getTile(newRow, newCol).isTravelable():
                                ## track the ant's position
                                if (newRow, newCol) not in worldCopy:
                                    worldCopy[(newRow, newCol)] = []
                                worldCopy[(newRow, newCol)].append(ant)

                    ## clear all ants on tile
                    world.getTile(row, col).clearAnts()
                         
                    
        for position, antList in worldCopy.items():
            row, col = position
            world.getTile(row, col).updateAnts(antList)
            
        anim.update(world)                           
        iters += 1
    anim.done()

class tile(object):
    """
    a tile represents a location in the worldMap
    """

    def __init__(self, numFood = 0, numAnts = 0, isTravelable = True, isHome = False):
        """
        initialize the tile with (row, col) position, amount of food, number of ants,
        whether the tile is traversable, and whether the tile is the home tile

        numFood: an integer indicating the number of food units present
        numAnts: an integer indicating the number of ants present
        isTravelable: a boolean indicating whether the tile can be travelled
        isHome: a boolean indicating whether the tile is the home tile
        """
        self.numFood = numFood
        self.arrAnts = []
        self.isOpen = isTravelable
        self.isHome = isHome

    def setNumFood(self, newFood):
        self.numFood = newFood

    def takeFood(self):
        self.numFood -= 1

    def addFood(self):
        self.numFood += 1

    def getAmountOfFood(self):
        return self.numFood

    def getAnts(self):
        return self.arrAnts

    def addAnt(self, antObj):
        self.arrAnts.append(antObj)

    def removeAnt(self, antObj):
        self.arrAnts.remove(antObj)

    def clearAnts(self):
        self.arrAnts = []
        
    def updateAnts(self, inputList):
        self.arrAnts = inputList

    def isTravelable(self):
        return self.isOpen

    def setIsTravelable(self, newBlocked):
        self.isOpen = newBlocked
        
    def isAntHome(self):
        return self.isHome    

    def setIsHome(self, newHome):
        self.isHome = newHome
    
class worldMap(object):
    """
    A worldMap represents a world

    A worldMap has a width and a height and contains (width*height) tiles
    """

    def __init__(self, width, height, blocked, food):
        """
        Initializes a worldMap with the specified
        width and height

        The percentage of tiles specified by blocked are set at
        random to be non-traversable

        The percentage of tiles specified by food are
        seeded at random with food in an amount between 1-5 units

        The anthill location is set at random

        width: an integer > 0
        height: an integer > 0
        blocked: a float 0.8 > float > 0.0
        food: a float 0.8 > float > 0.0

        Mapping of hash table values
        Open (i.e. traversable): 100
        Blocked: 101
        Home: 102
        Food: 1-5
         
        """
               
        self.width = width
        self.height = height

        self.dirs = dict([("NORTH",(-1,0)),("EAST",(0,1)),("SOUTH",(1,0)),("WEST",(0,-1))])


        ## create the map of the world             
        ## create an empty 2-dimensional array
        self.grid = Grid(height, width)

        ## populate the 2-dimensional array with open tiles
        for row in range(height):
            for col in range(width):
                self.grid.setAt(row, col, tile())
                
        ## set randomly selected tiles to be blocked
        assert blocked >= 0.0
        assert blocked <= 0.8
        for i in range(int(width*height*blocked)):
            val = random.choice(range(width*height))
            self.grid.getAt(val/width, val%width).setIsTravelable(False)

        ## set randomly selected tiles to be seeded with food
        assert food >= 0.0
        assert blocked <= 0.8
        for i in range(int(width*height*food)):
            val = random.choice(range(width*height))
            ## check that tile isn't blocked before seeding with food
            if self.grid.getAt(val/width, val%width).isTravelable():
                self.grid.getAt(val/width, val%width).setNumFood(random.choice(range(1,6)))

        ## set the anthill location
        while(True): 
            home = random.choice(range(width*height))
            ## check that tile isn't blocked before setting to be home
            currTile = self.grid.getAt(home/width, home%width)
            if currTile.isTravelable() and currTile.getAmountOfFood() == 0:
                self.grid.getAt(home/width, home%width).setIsHome(True)
                break;

        ## create a map of the ants in the world
        ## create an empty 2-dimensional array populated with empty lists
        self.antGrid = [[[] for col in xrange(width)] for row in xrange(height)]
      
    def getWorldWidth(self):
        return self.width

    def getWorldHeight(self):
        return self.height

    def getTile(self, row, col):
        """
        return the tile at the specified position
        """
        return self.grid.getAt(row, col)

    def getSurroundings(self, row, col):
        """
        return a hash table of the Surroundings of the position
        specified by the (row, col) position
        """

        output = {}
        output["Current"] = self.grid.getAt(row, col)

        ## iterate through each possible direction (NORTH, EAST, SOUTH, WEST)
        for direction, move in self.dirs.items():
            newRow = row + move[0]
            newCol = col + move[1]
            ## validate that the new position is in the world
            if newRow >= 0 and newRow < self.height and newCol >= 0 and newCol < self.width:     
                output[direction] = self.grid.getAt(newRow, newCol)
            ## dragons be here: off the map, set to blocked tile with 0 food, 0 ant
            else:                           
                output[direction] = tile(0,0,False)

        return output
              
class antSurroundings(object):
    """
    Surroundings represents the limited area that the
    ant can sense. Tiles to the North, East, South, West, Current
    """

    def __init__(self,startTiles):
        """
        Surroundings is initialized with a set of starting tiles
        """
        self.tileHash = startTiles

    def updatePosition(self, newHash):
        """
        updatePosition updates the hash table based on the
        new hash table of tiles passed as a parameter
        """
        self.tileHash = newHash

    def getCurrentTile(self):
        return self.tileHash["Current"]

    def getTile(self,direction):
        return self.tileHash[direction]

class Ant(object):
    """
    Represents an ant.

    The ant has an internal map and maps its position relative to
    the starting point
    """

    def __init__(self,inputPrintActions):
        """
        Ants are initialized with a 20x20 internal map. All tiles (except the
        starting tile) on the internal map are initially set to -100 which
        represents unexplored tiles. 

        The starting position of the ant is assumed to be (row, col)
        (9,9), the center of the internal map.

        Ants are initialized with zero food.

        Ants are intialzied with no route plan.

        The internal representation for ants is as follows:
        -100: unknown tile
        -101: travelable tile
        -102: untravelable tile
        -103: home tile
        0-127: food
        """

        self.foodKnown = False
        self.hasFood = False
        self.newFoodFound = False
        self.gridLen = 20
        self.receiveMap = False
        self.routePlan = []
        self.routePlanType = "UNKNOWN" 
        self.memo = {}
        self.row = 9
        self.col = 9
        self.homeRow = 9
        self.homeCol = 9
        self.foodLoc = (0,0)
        self.unexploredLoc = (0,0)
        self.foodLocSet = set()
        self.unexploredLocSet = set()
        self.exhaustedFoodLocSet = set()
        self.dirs = dict([("NORTH",(-1,0)),("EAST",(0,1)),("SOUTH",(1,0)),("WEST",(0,-1))])
        self.nReceives = 0
        self.printActions = inputPrintActions
        
        ## create a 2-dimensional grid to represent the world
        ## for the ant
        self.internalMap = Grid(self.gridLen, self.gridLen, -100) 

        self.internalMap.setAt(self.homeRow, self.homeCol, -103)    ##103 represents the home tile

    def getAntHasFood(self):
        return self.hasFood    

    def setAntHasFood(self, hasFood):
        """
        set the boolean value for whether the ant has food
        """
        self.hasFood = hasFood

    def antGather(self):
        """
        represents and ant collecting food
        """
        ## validate that the ant has no food
        assert self.hasFood == False
        self.hasFood = True

    def antDropoff(self):
        """
        represents and ant dropping off food
        """
        ## validate that the ant has food
        assert self.hasFood == True
        self.hasFood = False

    def setAntReceiveMap(self, mapFromOtherAnt):
        self.receiveMap = mapFromOtherAnt

    def getAntReceiveMap(self):
        return self.receiveMap

    def getPrintActions(self):
        return self.printActions

    def checkMap(self):
        """
        iterates through known food locations and know unexplored locations 
        to find the nearest tile of the respective type(food/unexplored).
        Stores the nearest location for use with A*.
        """
        self.foodLoc = (float("inf"),0)
        self.unexploredLoc = (float("inf"),0)
        for (row, col) in self.foodLocSet:
            ## A* hueristic: for each food location, calculate the distance form
            ## the current location 
            foodDistance = math.sqrt((self.row - row)**2 + (self.col - col)**2)
            if foodDistance < self.foodLoc[0]:
                self.foodLoc = (foodDistance,(row,col))

        for (row, col) in self.unexploredLocSet: 
            assert self.internalMap.getAt(row, col) == -100
            ## A* hueristic: for each unexplored location, calculate the distance form
            ## the current location
            unexploredDistance = math.sqrt((self.row - row)**2 + (self.col - col)**2)
            if unexploredDistance < self.unexploredLoc[0]:
                 self.unexploredLoc = (unexploredDistance,(row,col))

    def finalCheckMap(self):
        """
        iterates through the internal map to determine if there
        are any unexplored locations
        """
        for row in xrange(self.gridLen):
            for col in xrange(self.gridLen):
                if self.internalMap.getAt(row, col) == -101:
                    self.addUnexploredLocs(row,col)
                

    def updateMapPos(self, action):
        """
        updateMapPos takes a string value representing a movement.
        updateMapOs updates the relative location of the ant on the
        internal map
        """
        self.row = self.row + (self.dirs[action])[0]
        self.col = self.col + (self.dirs[action])[1]
        ## if the new position is within 5 tiles of the boundry of the ant's internal map
        ## then we increase the size of the internal map
        if self.row <= 5 or self.row > self.gridLen - 5 or self.col <= 5 or self.col > self.gridLen - 5:
            self.resize()

    def resizeInstanceVariables(self, rowDelta, colDelta):
        """
        resizeInstanceVaribles adjusts the instance variables
        so the instance variables stay consistent with the resized map
        """
        self.row += rowDelta
        self.col += colDelta
        self.homeRow += rowDelta
        self.homeCol += colDelta
        
        ## update the lists of food, unexplored, and exhausted
        self.foodLocSet = set(map(lambda x: (x[0] + rowDelta, x[1] + colDelta), self.foodLocSet))
        self.unexploredLocSet = set(map(lambda x: (x[0] + rowDelta, x[1] + colDelta), self.unexploredLocSet))
        self.exhaustedFoodLocSet = set(map(lambda x: (x[0] + rowDelta, x[1] + colDelta), self.exhaustedFoodLocSet))

    def resize(self):
        """
        resizes the ants internal map by adding 10 rows / 10 cols.
        """

        oldLength = self.gridLen
        self.gridLen += 20
        ## increase the ant's internal map. New tiles are set to -100 to represent unexplored
        self.internalMap.resize(10,10,-100)

        ## update the ant's position in the internal map
        self.resizeInstanceVariables(10,10)

            
    def shortest_path_search(self, startPos, fSuccessors, fIsGoal, goalObjective):
        """returns a list of the shortest path to a goal state.
        startPosition is a (row position,col position) tuple (i.e. coordinates) that
        specifies the starting location. SPS takes the staring state, the
        function successors, the function is_goal, and the variable goalType"""

        if fIsGoal([startPos], goalObjective):
            return [startPos]

        fail = []

        ## memoize paths to home and to food
        ## don't memoize paths to unexplored tiles as the unexplored tiles change 
        ## check to see if we've memoized the shortest path
        if (startPos, goalObjective) in self.memo:
            ## if the goal is to get home, we can use the memoized path   
            if goalObjective == "HOME": 
                return self.memo[(startPos, goalObjective)]
            ## if the goal is to find food, we need to confirm that
            ## we still believe there is food at the location
            elif goalObjective == "FOOD":
                ## pull the end position off the path and unpack the tuple
                endRow, endCol = (self.memo[(startPos, goalObjective)])[-1]
                ## if we believe there is still food at the end point
                ## we can take the memoized path
                if (endRow,endCol) in self.foodLocSet:
                    return self.memo[(startPos, goalObjective)]
                ## if there's no food left, delete the path
                else:
                    del self.memo[(startPos, goalObjective)]
                    
        
        ## algorithm for shortest_path_search
        explored = set([])
        frontier = Queue.PriorityQueue()
        frontier.put([1, [startPos]])

        while(frontier):
            ## path with lowest cost gValue is expanded first
            path = frontier.get()
            ## calculate the number of steps in the path to this point len(path[1])/2.
            numSteps, state = len(path[1])/2., (path[1])[-1]
            for (action,state) in fSuccessors(state).items():
                ## if our path intersects a known path for returning to the
                ## home tile, stop SPS and take the known path
                if goalObjective == "HOME" and (state, "HOME") in self.memo:
                    path2 = path[1] + [action] + self.memo[(state, "HOME")]
                    return path2 
                
                ## state is a tuple of row,col offsets
                if state not in explored:
                    explored.add(state)
                    ## path is in the form [(0,0),"NORTH",(-1,0),"WEST",(-1,-1),"NORTH",(-2,-1)]   
                    path2 = path[1] + [action, state]

                    ## calculations for A*: Calculate the euclidean distance from the
                    ## expansion location to the goal location. The distance will be used
                    ## to guide which location is expanded in subsequent steps 
                    rowOffset, colOffset = state
                    newRow = self.homeRow + rowOffset
                    newCol = self.homeCol + colOffset
                    if goalObjective == "HOME":
                        aStarDist = math.sqrt((rowOffset)**2 + (colOffset)**2)
                    elif goalObjective == "FOOD":
                        foodRow, foodCol = self.foodLoc[1]
                        aStarDist = math.sqrt((foodRow - newRow)**2 + (foodCol - newCol)**2)
                    else:
                        assert goalObjective == "UNKNOWN"
                        unexploredRow, unexploredCol = self.unexploredLoc[1]
                        aStarDist = math.sqrt((unexploredRow - newRow)**2 + (unexploredCol - newCol)**2)
                    ## the gValue is the number of steps  in the path so far plus the euclidean
                    ## distance to the goal 
                    gValPath = [numSteps + aStarDist, path2] 
                    if fIsGoal(path2, goalObjective):
                        ## only memoize food paths starting at the home tile. Note 1
                        if goalObjective == "HOME" or (goalObjective == "FOOD" and startPos == (self.homeRow, self.homeCol)): 
                            self.memo[(startPos, goalObjective)] = path2
                        return path2
                    else:
                        frontier.put(gValPath)
        return fail

    def fSuccessors(self, state):
        """returns a hash table of all possible sucessor states to the starting state
        key:value pairs consist of the action:new state. State is a row offset,col offset
        tuple (i.e. coordinates).""" 

        ## unpack the state tuple (row, col)
        ## this is the end state of the path we are checking
        rowOffset, colOffset = state 
        
        output = {}

        ## iterate through each direction:state pair in the hash table 
        for (direction,move) in self.dirs.items():
            ## update the row,col position by the move that corresponds to the direction
            newRowOffset = rowOffset + move[0]
            newColOffset = colOffset + move[1]
            newRow = self.homeRow + newRowOffset
            newCol = self.homeCol + newColOffset
            ## if the new row,col position isn't blocked
            ## and the new row,col position isn't unexplored
            ## add position to the hash table of possible successor states
            if (self.internalMap.getAt(newRow, newCol) != -100 and self.internalMap.getAt(newRow, newCol) != -102):
                output[direction] = (newRowOffset,newColOffset)
        return output

    ## helper function for shortest_path_search
    def fIsGoal(self, path, goalType):
        """returns a boolean whether the state represents a goal state
        there are 3 potential goals: home , food , and unknown
        """
    
        ## unpack the coordinate tuple
        rowOffset, colOffset = path[-1]
        row = self.homeRow + rowOffset
        col = self.homeCol + colOffset
        ## check if the path ends at the home tile
        if goalType == "HOME":
            return self.internalMap.getAt(row, col) == -103
        ## check if the path ends at food
        elif goalType == "FOOD":
            return (row, col) == self.foodLoc[1] 
        ## check if we're on a tile that can sense() an unexplored tile
        else:
            ## if not 1 or 2, goalType must be 0
            assert goalType == "UNKNOWN"
            output = False
            for move in self.dirs.values():
                newRow = row + move[0]
                newCol = col + move[1]
                if self.internalMap.getAt(newRow, newCol) == -100:
                    output = True
            return output


    def getAction(self,Surroundings):
        """
        returns a string value representing the ant's next action.
        Possible actions: GATHER, DROPOFF, HALT, NORTH, SOUTH, EAST, WEST
        """
        ## extract the current tile from Surroundings
        currTile = Surroundings.getCurrentTile()

        ## if there are other ants on the current tile, receive data from the other ants
        if len(currTile.getAnts()) > 0:
            for ant in currTile.getAnts():
                ## ants only communicate with other ants every 5th turn to reduce computational expense
                if ant != self and self.nReceives%2 == 0: 
                    self.receive(ant.send())

        ## sense the Surroundings
        self.sense(Surroundings)

        ## ant waits at home until it receives a map from a mature ant
        if not self.getAntReceiveMap():
            return "HALT"

        ## if the ant has a route plan, follow the plan
        if len(self.routePlan) > 0:
            action = self.routePlan.pop(0)
            self.updateMapPos(action)
            if self.getPrintActions():
                print action  
            return action
        
        ## if the ant has food and the ant is on the
        ## ant mound, drop off the food, DROPOFF
        ## the program decrements the ants food
        if self.getAntHasFood() and self.internalMap.getAt(self.row, self.col) == -103:
            if self.getPrintActions():
                print 'DROPOFF'  
            return "DROPOFF"
    
        ## if the ant doesn't have food and the current tile does have food
        ## set a path for home, then GATHER
        if currTile.getAmountOfFood() > 0 and not self.getAntHasFood() and self.internalMap.getAt(self.row, self.col) != -103:
            rowOffset = self.row - self.homeRow
            colOffset = self.col - self.homeCol
            path = self.shortest_path_search((rowOffset, colOffset), self.fSuccessors, self.fIsGoal, "HOME")
            self.routePlan = path[1::2]
            self.routePlanType = "HOME" 
            ## If there is only one unit of food delete the location as a food location
            if currTile.getAmountOfFood() == 1:
                self.foodLocSet.remove((self.row,self.col))                     
                self.exhaustedFoodLocSet.add((self.row, self.col))
                ## there's no longer food at the location so we no longer need the path
                if self.memo.has_key(((self.homeRow,self.homeCol), "FOOD")): 
                    del self.memo[((self.homeRow, self.homeCol), "FOOD")] 
            if self.getPrintActions():
                print "GATHER"  
            return "GATHER"

        ## if the ant is aware of a food location plot the shortest path to food
        if len(self.foodLocSet) > 0:
            assert not self.getAntHasFood()
            rowOffset = self.row - self.homeRow
            colOffset = self.col - self.homeCol
            path = self.shortest_path_search((rowOffset,colOffset),self.fSuccessors, self.fIsGoal, "FOOD")
            self.routePlan = path[1::2]
            self.routePlanType = "FOOD"
            ## update internal location and
            ## return the next action in the route plan
            action = self.routePlan.pop(0)
            self.updateMapPos(action)
            if self.getPrintActions():
                print action 
            return action
                    

        ## otherwise, the location of food is not known and the
        ## ant does not have a route, so plot a path to the
        ## nearest unexplored tile
        if len(self.foodLocSet) == 0 and len(self.unexploredLocSet) > 0:
            rowOffset = self.row - self.homeRow
            colOffset = self.col - self.homeCol
            path = self.shortest_path_search((rowOffset, colOffset),self.fSuccessors, self.fIsGoal, "UNKNOWN")
            self.routePlan = path[1::2]
            self.routePlanType = "UNKNOWN" 
            action = self.routePlan.pop(0)
            self.updateMapPos(action)
            if self.getPrintActions():
                print action  
            return action
        
        ## Otherwise we're done. No unexplored tiles
        else:
            if self.getPrintActions():
                print "HALT"
            self.finalCheckMap(); 
            return "HALT"
            
    def receive(self, externalData):
        """
        accepts an antMap from another ant and updates the ants
        internal map with the new information
        """
        self.nReceives += 1
        
        ## set the self.receiveMap field to true
        self.setAntReceiveMap(True)

        ## convert the 1-dimensional array to a 2-dimensional map
        extUnexploredLocSet, extFoodLocSet, extExhaustedFoodLocSet, extMap = self.reformGrid(externalData)
               
        ## maps are square. Extract the length of the sides   
        N = extMap.getRows()
        ## calculate the difference between the side of the internal map
        ## and the received map
        difference = N - self.gridLen
        halfDif = abs(difference/2)

        ## Size of internal map == size of external map
        ## update the internal map for the case where the size of the
        ## external map is the same as the size of the internal map
        ## Unexplored tiles on the internal map are updated where the
        ## tiles are known on the external map
    
        if difference == 0:
            for row in range(self.gridLen):
                for col in range(self.gridLen):
                    if self.internalMap.getAt(row, col) == -100 and extMap.getAt(row, col) != -100:
                        self.internalMap.setAt(row, col, extMap.getAt(row, col))
                        ## if the position was in the unexplored set, remove as it is no longer unexplored
                        self.unexploredLocSet.discard((row,col))

            self.sychronizeData(extUnexploredLocSet, extFoodLocSet, extExhaustedFoodLocSet)
                         
        ## Size of external map > size of internal map
        ## update for the case where the size of the
        ## external map greater than the size of the internal map
        ## Unexplored tiles on the internal map are updated where the
        ## tiles are known on the external map. After the update,
        ## the external map replaces the internal map

        elif difference > 0:
                      
            ## Copy information from internal map into external Grid
            for row in range(halfDif, halfDif + self.gridLen):
                for col in range(halfDif, halfDif + self.gridLen):
                    ## for explored tiles on the internal map, overwrite the external map
                    if not self.internalMap.getAt(row - halfDif, col - halfDif) == -100:
                        extMap.setAt(row, col, self.internalMap.getAt(row - halfDif, col - halfDif))
                                                
            ## overwite internal map with external map
            self.internalMap = extMap
            self.gridLen = N
            self.resizeInstanceVariables(halfDif, halfDif)
            self.sychronizeData(extUnexploredLocSet, extFoodLocSet, extExhaustedFoodLocSet)
            
        ## Size of internal map > size of external map
        ## update for the case where the size of the
        ## internal map is greater than the size of the external map
        ## Unexplored tiles on the internal map are updated where the
        ## tiles are known on the external map. 

        else:
            for row in range(halfDif, halfDif + N):
                for col in range(halfDif, halfDif + N):
                    if self.internalMap.getAt(row, col) == -100 and extMap.getAt(row - halfDif, col - halfDif) != -100:
                        self.internalMap.setAt(row, col, extMap.getAt(row - halfDif, col - halfDif))    
                     
            extUnexploredLocSet = set(map(lambda x: (x[0] + halfDif, x[1] + halfDif), extUnexploredLocSet))
            extFoodLocSet = set(map(lambda x: (x[0] + halfDif, x[1] + halfDif), extFoodLocSet))
            extExhaustedFoodLocSet = set(map(lambda x: (x[0] + halfDif, x[1] + halfDif), extExhaustedFoodLocSet))
            self.sychronizeData(extUnexploredLocSet, extFoodLocSet, extExhaustedFoodLocSet)
            

    def checkRoute(self, location):
        """
            checkRoute is needed to check and clear the current route plan.
            checkRoute is called when data is received from other ant
            that indicates that the closest food location is exhausted
        """
        if location == self.foodLoc[1]:
            rowOffset, colOffset = self.routePlan[-1]
            endRow = self.homeRow + rowOffset
            endCol = self.homeCol + colOffset
            if (endRow, endCol) == location:
                self.routePlan = []

    def sychronizeData(self,extUnexploredLocSet, extFoodLocSet, extExhaustedFoodLocSet):
        """
            sychronize data integrates the date from an external ant
        """
        
        self.exhaustedFoodLocSet.union(extExhaustedFoodLocSet)
        self.foodLocSet.union(extFoodLocSet)
        self.unexploredLocSet.union(extUnexploredLocSet)
        ## filter out locations that have already been explored 
        self.unexploredLocSet = set(filter(lambda x: self.internalMap.getAt(x[0], x[1]) == -100,self.unexploredLocSet))
        ## filter out food locations that are known to be exhausted
        self.foodLocSet = set(filter(lambda x: x not in self.exhaustedFoodLocSet, self.foodLocSet))
        ## delete the current routePlan if the new data shows food has been exhausted at the current route plan destination
        if self.foodLoc[1] in extExhaustedFoodLocSet and self.routePlanType == "FOOD":
            self.routePlan = [] 

                            
    def reformGrid(self, arr):
        """
        Ants transmit the following data structure
        Integer specifying the length of foodLocSet
        Integer specifying the length of the unexploredLocSet
        
        A 1-dimensional array and the length of the rows
        and returns a 2-dimensional array.

        Assumes that there are an equal number of rows and cols
        """

        offset1, offset2, offset3, offset4 = arr[0], arr[1], arr[2], arr[3] 
       
        
        externalUnexploredLocSet = arr[offset1:offset2]
        externalFoodLocSet = arr[offset2:offset3]
        externalExhaustedFoodLoc = arr[offset3:offset4] 
        externalMap = arr[offset4:] 

        ## internal maps are square. the square root of the array
        ## gives us the length of the rows
        N = int(math.sqrt(len(externalMap)))
        
        output = Grid(N,N)

        for row in xrange(N):
            for col in xrange(N):
                output.setAt(row, col, externalMap[row*N + col])

        return (externalUnexploredLocSet, externalFoodLocSet, externalExhaustedFoodLoc, output) 

    def send(self):
        """
        unroll the internal map into a 1-dimensional array for transmital
        """

        unexploredLen = len(self.unexploredLocSet)
        foodLen = len(self.foodLocSet)
        exhaustedFoodLen = len(self.exhaustedFoodLocSet)
        offset1 = 4
        offset2 = offset1 + unexploredLen
        offset3 = offset2 + foodLen
        offset4 = offset3 + exhaustedFoodLen

        output = [offset1, offset2, offset3, offset4]
        
        return output + \
               list(self.unexploredLocSet) +  \
               list(self.foodLocSet) + \
               list(self.exhaustedFoodLocSet) + \
               [self.internalMap.getAt(row,col) for row in xrange(self.gridLen) for col in xrange(self.gridLen)]

    def addUnexploredLocs(self, row, col):
        for move in self.dirs.values():
            newRow = row + move[0]
            newCol = col + move[1]
            if self.internalMap.getAt(newRow, newCol) == -100:
                self.unexploredLocSet.add((newRow, newCol))
            
            
       
    def sense(self, Surroundings):
        """
        sense takes a Surroundings object as input, examines the Surroundings,
        and updates the internal map
        """
        ## handle the corner case where the food on the current tile was taken since the last turn
        if self.internalMap.getAt(self.row, self.col) != -103 and Surroundings.getCurrentTile().getAmountOfFood() == 0 and \
        (self.row, self.col) in self.foodLocSet:
            self.exhaustedFoodLocSet.add((self.row, self.col))
            self.foodLocSet.remove((self.row, self.col))

        ## iterate through each possible direction
        for direction, move in self.dirs.items():
            sensedTile = Surroundings.getTile(direction)
            ## calculate the relative position for the sensed tile
            newRow = self.row + move[0]
            newCol = self.col + move[1]

            ## check for food, home tile always ignored 
            if self.internalMap.getAt(newRow, newCol) != -103 and sensedTile.getAmountOfFood() > 0:
                self.foodLocSet.add((newRow,newCol))
                ## note 2
                if self.routePlanType == "UNKNOWN":
                    self.routePlan = [] 

            ## unpdate map for travelable/untravelable, home tile always ignored 
            if self.internalMap.getAt(newRow, newCol) == -103:
                pass
            elif sensedTile.isTravelable():
                self.internalMap.setAt(newRow, newCol, -101)
                self.unexploredLocSet.discard((newRow,newCol))
            else:
                self.internalMap.setAt(newRow, newCol, -102)
                self.unexploredLocSet.discard((newRow,newCol))


        ##
        for move1 in self.dirs.values():
            newRow1 = self.row + move1[0]
            newCol1 = self.col + move1[1]
            if self.internalMap.getAt(newRow1, newCol1) == -101:
                self.addUnexploredLocs(newRow1, newCol1)

        ## after the ant finishes sensing, update whether food is known
        self.checkMap()
       

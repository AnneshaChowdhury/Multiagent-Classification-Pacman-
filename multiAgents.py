# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        newfoodList = newFood.asList();
        newGhostPosition = successorGameState.getGhostPositions()
        
        foodDist = [manhattanDistance(food, newPos) for food in newfoodList]
        ghostDistance = [manhattanDistance(ghost, newPos) for ghost in newGhostPosition]
        length = len(foodDist)
        if currentGameState.getPacmanPosition() == newPos:
            return -1000000

        for ghosdis in ghostDistance:
           if ghosdis < 1:
                return -1000000

        if length == 0:
            return 1000000
        else:
            minfoodDist = min(foodDist)
            maxfoodDist = max(foodDist)

        return 1000/sum(foodDist) + 10000/len(foodDist)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        bestScore,bestMove=self.maxFunction(gameState,self.depth)

        return bestMove

    def maxFunction(self,gameState,depth):
        if gameState.isWin() or gameState.isLose() or depth==0:
          return self.evaluationFunction(gameState), "none"

        actions=gameState.getLegalActions()
        
        scores = [self.minFunction(gameState.generateSuccessor(self.index,action),1, depth) for action in actions]
        bestIndices = [index for index in range(len(scores)) if scores[index] == max(scores)]
        bestScore=max(scores)
        
        return max(scores),actions[bestIndices[0]]

    def minFunction(self,gameState,agent, depth):  
        if depth==0 or gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState), "none"
        actions=gameState.getLegalActions(agent) 
        scores=[]
        if(agent!=gameState.getNumAgents()-1):
          scores =[self.minFunction(gameState.generateSuccessor(agent,action),agent+1,depth) for action in actions]
        else:
          scores =[self.maxFunction(gameState.generateSuccessor(agent,action),(depth-1))[0] for action in actions]
        
        worstIndices = [index for index in range(len(scores)) if scores[index] == min(scores)]
        minScore=min(scores)
        
        return minScore, actions[worstIndices[0]]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -float("inf")
        beta = float("inf")
        bestScore,bestMove=self.maxFunction(gameState,self.depth,alpha, beta)

        return bestMove

    def maxFunction(self,gameState,depth, alpha, beta):
        if depth==0 or gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState), "none"
        alpha = -float("inf")
        beta = float("inf")

        actions=gameState.getLegalActions()
        
        scores = [self.minFunction(gameState.generateSuccessor(self.index,action),1, depth, alpha, beta) for action in actions]
        bestIndices = [index for index in range(len(scores)) if scores[index] == max(scores)]
        bestScore=max(scores)
        if bestScore >= beta:
            return max(scores),actions[bestIndices[0]]
        alpha = max(alpha, bestScore)
        return bestScore, actions[bestIndices[0]]

    def minFunction(self,gameState,agent, depth, alpha, beta):  
        if depth==0 or gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState), "none"
        actions=gameState.getLegalActions(agent) 
        scores=[]
        if(agent!=gameState.getNumAgents()-1):
          scores =[self.minFunction(gameState.generateSuccessor(agent,action),agent+1,depth, alpha, beta) for action in actions]
        else:
          scores =[self.maxFunction(gameState.generateSuccessor(agent,action),(depth-1))[0] for action in actions]
        
        worstIndices = [index for index in range(len(scores)) if scores[index] == min(scores)]
        minScore=min(scores)
        if minScore <= alpha:
            return minScore, actions[worstIndices[0]]
        beta = min(beta, worstScore)
        return minScore, actions[worstIndices[0]]
    	
        
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectiminimax(gameState, self.depth, 0)[1]

    def expectiminimax(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), "None"
            
        else:
            if agentIndex == gameState.getNumAgents() - 1:
                depth = depth - 1
            if agentIndex == 0:
                maxAlpha = -100000000
            else:
                maxAlpha = 0
            maxAction = ''
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            actions = gameState.getLegalActions(agentIndex)
            l = len(actions)
            for action in actions:
                r = self.expectiminimax(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)
                if agentIndex == 0:
                    if r[0] > maxAlpha:
                        maxAlpha = r[0]
                        maxAction = action
                else:
                    maxAlpha += 1.0/l * r[0]
                    maxAction = action
            return (maxAlpha, maxAction)
       
        
    
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
"""better = betterEvaluationFunction
currentpos = currentGameState.getPacmanPosition()
currentfood = currentGameState.getPacmanFood()
ghostpositions = currentGameState.getGhostPosition()
capsulepos = currentGameState.getCapsules()
for ghostposition in ghostpos
    ghostDistance = util.manhattanDistance(pacmanPosition, ghostPosition)
for foodposition in foodpos
    foodDistance = util.manhattanDistance(pacmanPosition, foodPosition)
for capsuleposition in capsulepos
    capsuleDistance = util.manhattanDistance(pacmanPosition, capsulePosition)"""


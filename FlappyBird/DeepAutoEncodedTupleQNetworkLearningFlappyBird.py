import tensorflow as TensorFlow
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as Game
import random
import numpy as Numpy
from collections import deque

GAME = 'bird'
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 1000
EXPLORE = 1000000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1

CODE_SIZE = 16

def CodeConverter(codeBatch, coderBaseline):
    result = []
    for i in range(BATCH):
        number = 0
        for j in range(CODE_SIZE):
            number *= 2
            if codeBatch[i][j] > 0.5:
                number += 1
        result.append(number)
    return result

def Train():
    interactiveSession = TensorFlow.InteractiveSession()
    inputStates, code, loss = DAE.CreateAutoEncoder()
    train_step = TensorFlow.train.RMSPropOptimizer(1e-5).minimize(loss)
    tupleNetwork = [dict(), dict()]
    coderBaseline = Numpy.zeros((ACTIONS, 64))
    coderLock = False
    usedCodes = set()

    gameState = Game.GameState()

    coderRecordDqueue = deque();
    gameRecord = deque();
    scores = deque();
    currentScore = 0.;
    realScore = 0.
    
    globalMax = 0;
    gameCount = 1;

    interactiveSession.run(TensorFlow.global_variables_initializer())
    saver = TensorFlow.train.Saver(max_to_keep=100)

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = Numpy.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = gameState.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    currentStates = Numpy.stack((x_t, x_t,x_t, x_t), axis=2)
    currentStateCode = DAE.CodeConverter(code.eval(feed_dict={inputStates: [Numpy.reshape(currentStates, (25600))]})[0], coderBaseline, coderLock)

    

    t = 1
    epsilon = INITIAL_EPSILON
    while True:
        actions = Numpy.zeros([ACTIONS])
        actionIndex = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                actionIndex = random.randrange(ACTIONS)
                actions[random.randrange(ACTIONS)] = 1
            else:
                if not currentStateCode in tupleNetwork[0]:
                    tupleNetwork[0][currentStateCode] = 0
                if not currentStateCode in tupleNetwork[1]:
                    tupleNetwork[1][currentStateCode] = 0
                if tupleNetwork[0][currentStateCode] >= tupleNetwork[1][currentStateCode]:
                    actions[0] = 1
                else:
                    actions[1] = 1
                print("State: ", currentStateCode ," Select: ", actions[1], " Value: ", tupleNetwork[0][currentStateCode], ", ", tupleNetwork[1][currentStateCode])
        else:
            actions[0] = 1 # do nothing

        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        coloredAfterState, reward, terminal = gameState.frame_step(actions)
        afterState = cv2.cvtColor(cv2.resize(coloredAfterState, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, afterState = cv2.threshold(afterState, 1, 255, cv2.THRESH_BINARY)
        afterState = Numpy.reshape(afterState, (80, 80, 1))
        afterStates = Numpy.append(afterState, currentStates[:, :, :3], axis=2)
        afterStateCode = DAE.CodeConverter(code.eval(feed_dict={inputStates: [Numpy.reshape(afterStates, (25600))]})[0], coderBaseline, coderLock)

        currentScore += reward
        realScore += int(reward)

        # store the transition in Dequeue
        gameRecord.append((currentStateCode, actions, reward, afterStateCode, terminal))
        coderRecordDqueue.append(Numpy.reshape(currentStates, (25600)));
        if len(coderRecordDqueue) > REPLAY_MEMORY:
            coderRecordDqueue.popleft();

        if terminal or len(gameRecord) >= 10000:
            if len(gameRecord) >= 10000:
                gameState = Game.GameState()
            numpyScores = Numpy.array(scores);
            #print("Game", gameCount, " score: ", currentScore, " t: ", t);
            #with open('traningResult', 'a') as file:
            #    file.writelines(str(t) + "\t" + str(realScore) + "\t" + str(currentScore) + "\n")
            for i in range(0, len(gameRecord)):
                stateCode, actions, reward, aStateCode, terminal = gameRecord.pop()
                if t > OBSERVE*10:
                    action = 0 if actions[0] > 0.5 else 1
                    if not stateCode in tupleNetwork[action]:
                        tupleNetwork[action][stateCode] = 0
                    oldValue = tupleNetwork[action][stateCode]
                    if terminal:
                        newValue = -1
                    else:
                        if not aStateCode in tupleNetwork[0]:
                            tupleNetwork[0][aStateCode] = 0
                        if not aStateCode in tupleNetwork[1]:
                            tupleNetwork[1][aStateCode] = 0
                        if tupleNetwork[0][aStateCode] >= tupleNetwork[1][aStateCode]:
                            newValue = GAMMA * tupleNetwork[0][aStateCode] + reward
                        else:
                            newValue = GAMMA * tupleNetwork[1][aStateCode] + reward
                    tupleNetwork[action][stateCode] = oldValue + 0.0025 * (newValue - oldValue);


            scores.append(currentScore);
            if len(scores) > 100:
                scores.popleft();
            if currentScore > globalMax:
                globalMax = currentScore
            currentScore = 0.;
            realScore = 0.
            #if gameCount % 100 == 0:
            #    print("=========================");
            #    print("AverageScore: ", Numpy.mean(numpyScores), ", Deviation: ", Numpy.std(numpyScores));
            #    print("MaxScore: ", Numpy.max(scores));
            #    print("GlobalMaxScore: ", globalMax);
            #    print("=========================");
            gameCount = gameCount + 1;
        if t > OBSERVE:
            coderBatch = random.sample(coderRecordDqueue, BATCH)
            if t % 1000 == 0:
                print("step %d, loss %g"%(t, loss.eval(feed_dict={inputStates: coderBatch})))
                #if loss.eval(feed_dict={inputStates: coderBatch}) < 2000:
                #    coderLock = True
                #print(CodeConverter(code.eval(feed_dict={inputStates: coderBatch}), coderBaseline))
                #print("Used Codes: ", len(usedCodes))
            if not coderLock:
                train_step.run(feed_dict={inputStates: coderBatch})

        # update the old values
        usedCodes.add(currentStateCode)
        currentStates = afterStates
        currentStateCode = afterStateCode
        t += 1

def PlayGame():
    Train()

def main():
    PlayGame()

if __name__ == "__main__":
    main()

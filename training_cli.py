#!/usr/bin/env python3
import numpy as np

def cli(model, vm, X, Y):
    np.random.seed(5)
    np.random.shuffle(X)
    np.random.seed(5)
    np.random.shuffle(Y)
    """X = X[:400]
    Y = Y[:400]"""
    X = np.expand_dims(X, axis=2)
    """train_size = 80 * len(X) // 100
    test_size = 10 * len(X) // 100
    validation_size = test_size
    X_train, X_test, X_validation = X[:train_size], X[train_size:-validation_size], X[train_size + test_size:]
    Y_train, Y_test, Y_validation = Y[:train_size], Y[train_size:-validation_size], Y[train_size + test_size:]"""
    train_size = 80 * len(X) // 100
    test_size = 20 * len(X) // 100
    X_train, X_test, = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    print(len(X_train), len(X_test))
    while True:
        cmd = input("Enter epoch:")
        if cmd == 'stop':
            break
        elif cmd == "evaluate":
            print(model._model.evaluate(X_test, Y_test))
        elif cmd == "eval_all":
            print(model._model.evaluate(X, Y))
        else:
            vm.model.train(X_train, Y_train, batch_size=64, epoch=int(cmd))
            vm.model.save("word_test.model")
    #print(model._model.evaluate(X_validation, Y_validation))



if __name__ == '__main__':
    main()

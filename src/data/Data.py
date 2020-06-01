class Data:

    sets = ()

    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
        sets = train_set, test_set

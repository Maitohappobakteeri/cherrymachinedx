class Configuration:
    def __init__(self, args):
        for arg in vars(args):
            self.__dict__[arg] = getattr(args, arg)

        # self.dataDir = "cherry-data/"
        self.dataDir = "dataset/images/"

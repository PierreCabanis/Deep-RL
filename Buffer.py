class Buffer:
    def __init__(self, taille_buffer):
        self.content = [None]*taille_buffer
        self.index = 0
        self.taille = taille_buffer

    def append(self, o):
        self.content[self.index] = o
        self.index = (self.index + 1) % self.taille


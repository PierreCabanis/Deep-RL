from random import sample


class Buffer:
    def __init__(self, taille_buffer):
        self.content = []
        self.index = 0
        self.taille = taille_buffer

    def append(self, o):
        if self.index < len(self.content):
            self.content[self.index] = o
        else:
            self.content.append(o)

        self.index = (self.index + 1) % self.taille

    def get_batch(self, batch_size):
        return sample(self.content, batch_size)

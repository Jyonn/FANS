class Order:
    class Col:
        def __init__(self, col):
            if isinstance(col, list):
                self.name = col[0]
                self.attrs = col[1:]
            else:
                self.name = col
                self.attrs = []

    def __init__(self, order):
        self.order = [Order.Col(col) for col in order]

    def __getitem__(self, index):
        return self.order[index]

    def __len__(self):
        return len(self.order)

    def __iter__(self):
        return iter(self.order)

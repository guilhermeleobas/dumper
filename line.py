from collections.abc import Sequence

class Block(Sequence):
    def __init__(self, *args):
        self.args = list(args)

    def __str__(self) -> str:
        final = '\n'.join(map(str, self.args))
        if self.args[-1] == Line.new_line():
            return final.removesuffix('\n')
        return final

    def to_string(self):
        return str(self)

    def __getitem__(self, idx):
        return self.args[idx]

    def __len__(self):
        return len(self.args)

    def __add__(self, other):
        assert isinstance(other, (Line, Block))
        if isinstance(other, Line):
            self.args.append(other)
        elif isinstance(other, Block):
            self.args.extend(other.args)
        return self


class Line(str):
    """
    """
    @classmethod
    def new_line(cls):
        return cls('\n')

    def __init__(self, line: str) -> None:
        self._line = line

    def __str__(self) -> str:
        return self._line

    def __add__(self, other):
        return f'{str(self)}\n{str(other)}'


class Comment(Line):

    @classmethod
    def new_line(cls):
        return cls('\n')

    def __init__(self, line: str) -> None:
        super().__init__(line)

    def __str__(self) -> str:
        return f'# {self._line}'

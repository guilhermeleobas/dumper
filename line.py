
class Block:
    def __init__(self, *args):
        self.args = list(args)

    def __str__(self) -> str:
        final = ''
        for arg in self.args:
            final += str(arg) + '\n'
        return final

    def resolve(self):
        return str(self).strip()

    def __add__(self, other):
        assert isinstance(other, (Line))
        self.args.append(other)
        return self


class Line(str):
    """
    """
    def __init__(self, line: str) -> None:
        self._line = line

    def __str__(self) -> str:
        return self._line

    def __add__(self, other):
        return f'{str(self)}\n{str(other)}'


class Comment(Line):
    def __init__(self, line: str) -> None:
        super().__init__(line)

    def __str__(self) -> str:
        return f'# {self._line}'

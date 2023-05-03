# Lets define a set of operators.

# We should use ADT to implement a simple and self-descriptive interface
# which encodes an ordered set of operators to apply on each of generation.

# Though the other way is to define a factory with 'frozen' set of operators.

# For now we will use a factory pattern.


def create_operators_set(*operators):
    pass


class Operator:
    def __init__(self):
        pass

    def apply(self, generation):
        raise NotImplementedError("This function has to be implemented")

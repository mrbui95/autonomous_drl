from numpy import sqrt


class Point:
    def __init__(self, x=0, y=0):
        self.__x = x
        self.__y = y

    def get_point(self):
        return (self.__x, self.__y)

    def get_dis_to_point(self, point):
        x, y = point.get_point()
        return sqrt((x - self.__x) ** 2 + (y - self.__y) ** 2)

    def slope_with_point(self, point):
        x2, y2 = point.get_point()
        if x2 == self.__x:
            return float("inf")
        else:
            return (y2 - self.__y) / (x2 - self.__x)

    def __eq__(self, value):
        if type(value) is Point:
            x, y = value.get_point()
        elif type(value) is tuple:
            x, y = value
        else:
            raise ValueError(
                "the unknown supported type value {} {}".format(type(value), value)
            )
        return x == self.__x and y == self.__y

    def __str__(self):
        return str(self.__x) + ", " + str(self.__y)

    def __hash__(self):
        return hash((self.__x, self.__y))

    def __lt__(self, value):
        return self.__x < value.get_point()[0]

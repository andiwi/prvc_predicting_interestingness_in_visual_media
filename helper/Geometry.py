class Rect(object):
    def __init__(self, x,y,w,h):
        '''
        Stores left x, bottom y, width, height of a rectangle
        :param x: 
        :param y: 
        :param w: 
        :param h: 
        '''
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def area(self):
        '''
        calculates area of rectangle
        :return: area
        '''
        return ((self.x + self.w) - self.x) * ((self.y + self.h) - self.y)

    @staticmethod
    def intersection(a, b):
        '''
        intersects two rectangles
        rectangles must overlap
        :param a: Rect
        :param b: Rect
        :return: Rect
        '''
        x1 = max(a.x, b.x)
        y1 = max(a.y, b.y)
        x2 = min(a.x + a.w, b.x + b.w)
        y2 = min(a.y + a.h, b.y + b.h)
        w = x2-x1
        h = y2-y1
        return Rect(x1,y1,w,h)

    @staticmethod
    def overlap(a, b):
        '''Overlapping rectangles overlap both horizontally & vertically
        '''
        return Rect.range_overlap(a.x, a.x+a.w, b.x, b.x+b.w) and Rect.range_overlap(a.y, a.y+a.h, b.y, b.y+b.h)

    @staticmethod
    def range_overlap(a_min, a_max, b_min, b_max):
        '''Neither range is completely greater than the other
        '''
        return (a_min <= b_max) and (b_min <= a_max)

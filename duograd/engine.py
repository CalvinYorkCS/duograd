import math

class Value:
    """ Stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._grad = 0

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op 

    @property # lazy loading grad as Value
    def grad(self):
        if isinstance(self._grad, (int, float)):
            self._grad = Value(self._grad, _op='grad_init')
        return self._grad
    
    @grad.setter
    def grad(self, value):
        self._grad = value

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data + other.data, (self, other), '+')

        def _backward(): 
            self.grad = self.grad + out.grad 
            other.grad = other.grad + out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        # print("in mul")
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data * other.data, (self, other), '*') 

        def _backward(): 
            self.grad = self.grad + (other * out.grad)
            other.grad = other.grad + (self * out.grad)
        out._backward = _backward

        # print("out mul")
        return out
    
    def __pow__(self, other):
        # print("in pow")
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad = self.grad + (other * (self**(other-1)) * out.grad)
        out._backward = _backward

        # print("out pow")
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        x = self.data
        out = Value((math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)), (self,), 'Tanh')

        def _backward():
            self.grad = self.grad + ((1 - out**2) * out.grad)
        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()
        def create_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    create_topo(child) 
                topo.append(v)
        create_topo(self)

        self.grad = Value(1)
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad.data})"
    
if __name__ == '__main__':
    x = Value(3.0)
    f = x**3

    f.backward()
    print(f"First derivative (3*x^2 at x=3): {x.grad.data}") 
    # Should be 27.0

    x.grad.data = 0
    x.grad.backward()
    print(f"Second derivative (6*x at x=3): {x.grad.data}")
    # Should be 18.0
class Value:
    """ Stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = Value(0) # Treating grads as value objects as well now

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op 

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data + other.data, (self, other), '+')

        def _backward(): 
            self.grad = self.grad + out.grad # changed from += to + to allow __add__ method to apply to grads
            other.grad = other.grad + out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data * other.data, (self, other), '*') 

        def _backward(): 
            self.grad = self.grad + (other.data * out.grad)
            other.grad = self.grad + (self.data * out.grad)
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad = self.grad + ((other * self.data**(other-1)) * out.grad)
        out._backward = _backward

        return out

    def __relu__(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
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
        return f"Value(data={self.data}, grad={self.grad})"
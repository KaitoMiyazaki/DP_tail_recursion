def func():
    x = 3
    def value(v):
        nonlocal x
        x = v

    def add(y):
        return y+x
    x = 5
    return value, add

    
v,f = func()
print(f(4))
v(10)
print(f(4))
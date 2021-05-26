def func():
    x = 3
    def value(v):
        nonlocal x #束縛された変数の値をあとから更新．
        x = v

    def add(y):
        return y+x
    x = 5
    return value, add

    
v,f = func()
print(f(4))
v(10)
print(f(4))
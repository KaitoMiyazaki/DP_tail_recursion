from functools import wraps

def tail_recursion(func):
  firstcall = True
  params = ((), {})
  result = func

  @wraps(func)
  def wrapper(*args, **kwd):
    nonlocal firstcall, params, result
    params = args, kwd
    if firstcall:
      firstcall = False
      try:
        while result is func:
          result = func(*args, **kwd) # call wrapper
          args, kwd = params
      finally:
        firstcall = True
        return result
    else:
      return func

  return wrapper

@tail_recursion
def fact(n, acc=1):
  return fact(n-1, acc*n) if n > 0 else acc

print(fact(10))
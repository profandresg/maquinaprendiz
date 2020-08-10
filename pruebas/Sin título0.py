# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 13:30:30 2020

@author: amgonzalezv
"""

print(str(2 + 2))
print(str(50 - 5*6))
print(str((50 - 5*6) / 4))
print(str(8 / 5))  # division always returns a floating point number
17 / 3  # classic division returns a float
17 // 3  # floor division discards the fractional part
17 % 3;  # the % operator returns the remainder of the division

def fib(n):    # write Fibonacci series up to n
    """Print a Fibonacci series up to n."""
    a, b = 0, 1
    mylist=[]
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
        mylist.append(b)
    print('thend')
    return mylist
    
# Now call the function we just defined:
fibo=fib(100)
print(fibo)

#%% 
def ask_ok(prompt, retries=4, reminder='Please try again!'):
    while True:
        ok = input(prompt)
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no', 'nop', 'nope'):
            return False
        retries = retries - 1
        if retries < 0:
            raise ValueError('invalid user response')
        print(reminder)
        
ask_ok('Do you really want to quit?')


#%%
def f(a, L=None):
    if L is None:
        L = []
    L.append(a)
    return L

print(f(4))


#%% 

def parrot(voltage, state='a stiff', action='voom', type='Norwegian Blue'):
    print("-- This parrot wouldn't", action, end=' ')
    print("if you put", voltage, "volts through it.")
    print("-- Lovely plumage, the", type)
    print("-- It's", state, "!")

parrot(1000)                                          # 1 positional argument
parrot(voltage=1000)                                  # 1 keyword argument
parrot(voltage=1000000, action='VOOOOOM')             # 2 keyword arguments
parrot(action='VOOOOOM', voltage=1000000)             # 2 keyword arguments
parrot('a million', 'bereft of life', 'jump')         # 3 positional arguments
parrot('a thousand', state='pushing up the daisies')  # 1 positional, 1 keyword
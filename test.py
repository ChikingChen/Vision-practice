from functools import cmp_to_key

def compare(a, b):
    if a < b:
        return 1
    elif a == b:
        return 0
    else:
        return -1

lst = [3, 4, 5, 2, 3, 1]
print(list(set(lst)))

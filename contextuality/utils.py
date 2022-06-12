def sum_odd(arr):
    """
    Compute the maximum value of the sum of the given 
    array `arr` with odd number of elements first mutliplied
    by -1.
    """
    arr_neg = [a for a in arr if a < 0]
    if len(arr_neg) == 0:
        return sum(arr) - 2 * min(arr)

    arr_pos = [a for a in arr if a >= 0]
    s = sum(arr_pos) - sum(arr_neg)
    if len(arr_neg) % 2 == 0:
        return s + 2 * max(arr_neg)
    else:
        return s


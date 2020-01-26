import numpy as np
import pandas as pd

def divisible_by(arr, int1, int2):
    '''Returns an array of the integers in arr that are divisible without
    remainder by both int1 and int2.

    Parameters
    ----------
    arr: NumPy array
    int1, int2: int, int

    Returns
    -------
    NumPy array

    >>>divisible_by(np.array([0, 24, 3, 12, 18, 17]), 3, 4)
    np.array([0, 24, 12])
    '''
    # arr1 = numpy.remainder(arr, int1)
    # arr2 = numpy.remainder(arr, int2)
    # arr3 = np.where(arr1 == 0, True, False)
    # arr4 = np.where(arr2 == 0, True, False)
    # arr5 = arr3 * arr4
    # arr6 = arr * arr5
    # numpy
    # for e in arr6


if __name__=='__main__':


    '''Returns the numbers from 1 to 100 into list. But for
    multiples of three append "Fizz" instead of the number and for the multiples
    of five append "Buzz". For numbers which are multiples of both three and five
    append "FizzBuzz".

    The first five elements of the output list are:
        lst = [1, 2, "Fizz", 4, "Buzz", ....]

    Parameters
    ----------
    None

    Returns
    -------
    list
    '''

    # lst = list(range(1, 101))

    # for i, e in enumerate(lst):
    #     if e % 5 == 0 and e % 3 == 0:
    #         lst[i] = "FizzBuzz"
    #     elif e % 3 == 0:
    #         lst[i] = "Fizz"
    #     elif e % 5 == 0:
    #         lst[i] = "Buzz"
    
    # print(lst)

    '''Computes a list of the first 20 Fibonacci numbers.
    By definition, the first two numbers in the Fibonacci sequence are 0 and 1,
    and each subsequent number is the sum of the previous two.

    The first 10 Fibonnaci numbers: 0, 1, 1, 2, 3, 5, 8, 13, 21, and 34.

    Parameters
    ----------
    None

    Returns
    -------
    list
    '''
    lst = [0, 1]
    for i in range(2, 20):
        lst.append(lst[i - 1] + lst[i -2])
    
    print(lst)
    
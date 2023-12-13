from scipy.stats import wilcoxon
import numpy as np

def statisticalTests(a, b):
    def a12(lst1, lst2, rev=True):
        more = same = 0.0
        for x in lst1:
            for y in lst2:
                if x == y:
                    same += 1
                elif rev and x > y:
                    more += 1
                elif not rev and x < y:
                    more += 1
        return (more + 0.5 * same) / (len(lst1) * len(lst2))

    ##  Wilcoxon signed-rank test...
    res = wilcoxon(a, b)
    p_value = res.pvalue
    ##  Vargha-Delaney A^12 test...
    a12_value = a12(a, b)
    return p_value, a12_value

if __name__ == '__main__':
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([-1, -2, -3, -4, -5])
    p_value, a12_value = statisticalTests(a, b)
    print(f'p_value: {p_value}, a12_value: {a12_value}')


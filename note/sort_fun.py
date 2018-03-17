#coding:utf-8
import time
import numpy as np

#  装饰器
def fn_timer(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        res = fn(*args, **kwargs)
        print('{0} cost time {1}'.format(fn.__name__, time.clock() - start))
        return res
    return _wrapper

@fn_timer
def bubble_sort(l):
    """
    每次比较相邻的两个元素，每次遍历将最大或最小的元素固定位置
    :param l:
    :return:
    """
    for _ in range(len(l) - 1):
        for i in range(len(l) - 1):
            if l[i] > l[i + 1]:
                l[i], l[i + 1] =  l[i + 1], l[i]
    return l

@fn_timer
def select_sort(l):
    """
    每次选择一个最大或最小的元素固定位置
    :param l:
    :return:
    """
    for j in range(len(l) - 1):
        k = j
        for i in range(j + 1, len(l)):
            if l[i] < l[k]:
                k = i
        if j != k:
            l[j], l[k] = l[k], l[j]
    return l

@fn_timer
def insert_sort(l):
    """
    每次把前 i + 1 各元素排序
    :param l:
    :return:
    """
    for j in range(1, len(l)):
        for i in range(j, 0, -1):
            if l[i - 1] > l[i]:
                l[i - 1], l[i] = l[i], l[i - 1]
    return l

def quick_sort(l):
    """
    递归排列小于制定元素的list 和 大于指定元素的list,最后合并
    :param l:
    :return:
    """
    if len(l) == 1:
        return l
    elif len(l) == 0:
        return []
    else:
        pivot = l[0]
        left = quick_sort([x for x in l[1:] if x < pivot])
        right = quick_sort([x for x in l[1:] if x >= pivot])
        return left + [pivot] + right


def fn_call(fn, *args):
    start = time.clock()
    res = fn(*args)
    print('{0} cost time:{1}'.format(fn.__name__, time.clock() - start))
    return res

if __name__ == '__main__':
    l = np.random.rand(1,1000)
    bubble_sort(l)
    l = np.random.rand(1,1000)
    select_sort(l)
    l = np.random.rand(1,1000)
    insert_sort(l)
    l = np.random.rand(1,1000)
    fn_call(quick_sort, l)
__author__ = 'T'
import time
import mmap
from multi_proc.jsonmap import ObjectMmap
import random

def main():
    mm = ObjectMmap(-1, 1024*1024, access=mmap.ACCESS_WRITE, tagname='share_mmap')
    while True:
        time.sleep(2)
        length = random.randint(1, 100)
        p = range(length)
        mm.jsonwrite(p)

        print '*' * 30
        print mm.jsonread_master()

if __name__ == '__main__':
    main()
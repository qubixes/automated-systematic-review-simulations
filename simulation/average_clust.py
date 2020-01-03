#!/usr/bin/env python

import sys
from numpy import average


def data_avg(data):
    return average([x[1] for x in data])


def main(data_fp):
    with open(data_fp, "r") as fp:
        all_data = []
        for line in fp:
            data = [float(x) for x in line.split()]
            all_data.append(data)
        n_take_high = int(len(all_data)/2)
        sorted_data = sorted(all_data, key=lambda x: x[0], reverse=True)
        high_data = sorted_data[:n_take_high]
        low_data = sorted_data[n_take_high:]
        print(data_avg(all_data), data_avg(high_data), data_avg(low_data))


if __name__ == "__main__":
    main(sys.argv[1])

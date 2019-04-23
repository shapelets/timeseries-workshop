#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Shapelets.io
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
import time
import matplotlib.pyplot as plt


def sliding_dot_product(q, t):
    n = t.size
    m = q.size

    # Append t with n zeros
    ta = np.append(t, np.zeros(n))

    # Reverse Q
    qr = np.flip(q, 0)

    # Append qra
    qra = np.append(qr, np.zeros(2 * n - m))

    # Compute FFTs
    qraf = np.fft.fft(qra)
    taf = np.fft.fft(ta)

    # Compute the inverse FFT to the element-wise multiplication of qraf and taf
    qt = np.fft.ifft(np.multiply(qraf, taf))
    return qt[m:n]


def sliding_dot_product_stomp(q, t):
    n = t.size
    m = q.size

    # Append t with n zeros
    ta = np.append(t, np.zeros(n))

    # Reverse Q
    qr = np.flip(q, 0)

    # Append qra
    qra = np.append(qr, np.zeros(2 * n - m))

    # Compute FFTs
    qraf = np.fft.fft(qra)
    taf = np.fft.fft(ta)

    # Compute the inverse FFT to the element-wise multiplication of qraf and taf
    qt = np.fft.ifft(np.multiply(qraf, taf))
    return qt[m - 1:n]


def calculate_distance_profile(q, t, qt, a, sum_q, sum_q2, mean_t, sigma_t):
    n = t.size
    m = q.size

    b = np.zeros(n - m)
    dist = np.zeros(n - m)
    for i in range(0, n - m):
        b[i] = -2 * (qt[i].real - sum_q * mean_t[i]) / sigma_t[i]
        dist[i] = a[i] + b[i] + sum_q2
    return np.sqrt(np.abs(dist))


def compute_mean_std_for_query(Q):
    # Compute Q stats -- O(n)
    sumQ = np.sum(Q)
    sumQ2 = np.sum(np.power(Q, 2))
    return sumQ, sumQ2


def pre_compute_mean_std_for_TS(ta, m):
    na = len(ta)
    sum_t = np.zeros(na - m)
    sum_t2 = np.zeros(na - m)

    # Compute the stats for t
    cumulative_sum_t = np.cumsum(ta)
    cumulative_sum_t2 = np.cumsum(np.power(ta, 2))
    for i in range(na - m):
        sum_t[i] = cumulative_sum_t[i + m] - cumulative_sum_t[i]
        sum_t2[i] = cumulative_sum_t2[i + m] - cumulative_sum_t2[i]
    mean_t = np.divide(sum_t, m)
    mean_t2 = np.divide(sum_t2, m)
    mean_t_p2 = np.power(mean_t, 2)
    sigma_t2 = np.subtract(mean_t2, mean_t_p2)
    sigma_t = np.sqrt(sigma_t2)
    return sum_t, sum_t2, mean_t, mean_t2, mean_t_p2, sigma_t, sigma_t2


def pre_compute_mean_std_for_TS_stomp(ta, m):
    na = len(ta)
    # Compute the stats for t
    cumulative_sum_t = np.cumsum(ta)
    cumulative_sum_t2 = np.cumsum(np.power(ta, 2))
    sum_t = (cumulative_sum_t[m - 1:na] - np.concatenate(([0], cumulative_sum_t[0:na - m])))
    sum_t2 = (cumulative_sum_t2[m - 1:na] - np.concatenate(([0], cumulative_sum_t2[0:na - m])))
    mean_t = np.divide(sum_t, m)
    mean_t2 = np.divide(sum_t2, m)
    mean_t_p2 = np.power(mean_t, 2)
    sigma_t2 = np.subtract(mean_t2, mean_t_p2)
    sigma_t = np.sqrt(np.abs(sigma_t2))
    return sum_t, sum_t2, mean_t, mean_t2, mean_t_p2, sigma_t, sigma_t2


# Mueen’s algorithim for similarity search (MASS)
def mass(Q, T, a, meanT, sigmaT):
    # Z-Normalisation
    if np.std(Q) != 0:
        Q = (Q - np.mean(Q)) / np.std(Q)
    QT = sliding_dot_product(Q, T)
    sumQ, sumQ2 = compute_mean_std_for_query(Q)
    return calculate_distance_profile(Q, T, QT, a, sumQ, sumQ2, meanT, sigmaT)


def element_wise_min(pab, Iab, D, idx, ignore_trivial, m):
    for i in range(0, len(D)):
        # if it's a self-join, ignore trivial matches in [-m/2,m/2]
        if not ignore_trivial or (np.abs(idx - i) > m / 2.0):
            if D[i] < pab[i]:
                pab[i] = D[i]
                Iab[i] = idx
    return pab, Iab


def stamp(ta, tb, m):
    """
    Compute the Matrix Profile between time-series ta and tb.
    If ta==tb, the operation is a self-join and trivial matches are ignored.
    
    :param ta: time-series, np.array
    :param tb: time-series, np.array
    :param m: subsequence length
    :return: Matrix Profile, Nearest-Neighbor indexes
    """
    nb = len(tb)
    na = len(ta)
    pab = np.ones(na - m) * np.inf
    Iab = np.zeros(na - m)
    idxes = np.arange(nb - m + 1)

    sumT, sumT2, meanT, meanT_2, meanTP2, sigmaT, sigmaT2 = pre_compute_mean_std_for_TS(ta, m)

    a = np.zeros(na - m)
    for i in range(0, na - m):
        a[i] = (sumT2[i] - 2 * sumT[i] * meanT[i] + m * meanTP2[i]) / sigmaT2[i]

    ignore_trivial = np.atleast_1d(ta == tb).all()
    for idx in idxes:
        D = mass(tb[idx: idx + m], ta, a, meanT, sigmaT)
        if (ignore_trivial):
            minIdx = int(np.maximum(idx - m / 2.0, 0))
            maxIdx = int(np.minimum(idx + m / 2.0, len(D)))
            D[minIdx:maxIdx:1] = np.inf

        Iab[pab > D] = i
        pab = np.minimum(pab, D)
    return pab, Iab


def stampSJ(T, m):
    return stamp(T, T, m)


def stomp(T, T2, m):
    """
    Compute the Matrix Profile with self join for T

    :param T: time-series, np.array
    :param tb: time-series, np.array
    :param m: subsequence length
    :return: Matrix Profile, Nearest-Neighbor indexes
    """
    epsilon = 1e-10

    n = len(T)

    seq_l = n - m
    _, _, meanT, _, _, sigmaT, _ = pre_compute_mean_std_for_TS_stomp(T, m)

    pab = np.full(seq_l + 1, np.inf)
    Iab = np.zeros(n - m + 1)
    ignore_trivial = True
    for idx in range(0, seq_l):
        Q_std = sigmaT[idx] if sigmaT[idx] > epsilon else epsilon
        if idx == 0:
            QT = sliding_dot_product_stomp(T2[0:m], T).real
            QT_first = np.copy(QT)
        else:
            QT[1:] = QT[0:-1] - (T[0:seq_l] * T[idx - 1]) + (T[m:n] * T[idx + m - 1])
            QT[0] = QT_first[idx]

        # Calculate distance profile
        D = np.abs(2 * (m - (QT - m * meanT * meanT[idx]) / (Q_std * sigmaT)))
        D[D < epsilon] = 0.0
        if (ignore_trivial):
            # ignore trivial minimum and  maximum
            min_idx = int(np.maximum(idx - m / 2.0, 0))
            max_idx = int(np.minimum(idx + m / 2.0, len(D)))
            D[min_idx:max_idx:1] = np.inf

        Iab[pab > D] = idx
        np.minimum(pab, D, pab)

    np.sqrt(pab, pab)
    return pab, Iab


def stompSJ(T, m):
    """
    Compute the Matrix Profile with self join for T

    :param T: time-series, np.array
    :param tb: time-series, np.array
    :param m: subsequence length
    :return: Matrix Profile, Nearest-Neighbor indexes
    """
    return stomp(T, T, m)


def find_best_discord(profile):
    return np.argmax(profile)


def find_best_motif(profile):
    return np.argmin(profile)


# Quick Test
def test_stomp(ta, m):
    start_time = time.time()

    pab, iab = stomp(ta, m)
    print("Execution time: " + str(time.time() - start_time))
    plot_motif(ta, pab, iab, m)
    return pab, iab


# Quick Test
def test_stamp(ta, tb, m):
    start_time = time.time()

    pab, Iab = stamp(ta, tb, m)
    print("--- %s seconds ---" % (time.time() - start_time))

    plot_discord(ta, pab, Iab, m, )
    return pab, Iab


def plot_motif(ta, values, m):
    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(ta, linestyle='--', alpha=0.5)
    plt.xlim((0, len(ta)))

    print(np.argmax(values))

    plt.plot(range(np.argmin(values), np.argmin(values) + m), ta[np.argmin(values):np.argmin(values) + m], c='g',
             label='Top Motif')
    plt.plot(range(np.argmax(values), np.argmax(values) + m), ta[np.argmax(values):np.argmax(values) + m], c='r',
             label='Top Discord')

    plt.legend(loc='best')
    plt.title('Time-Series')

    plt.subplot(212)
    plt.title('Matrix Profile')
    plt.plot(range(0, len(values)), values, '#ff5722')
    plt.plot(np.argmax(values), np.max(values), marker='x', c='r', ms=10)
    plt.plot(np.argmin(values), np.min(values), marker='^', c='g', ms=10)

    plt.xlim((0, len(ta)))
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.tight_layout()

    plt.show()


def plot_discord(ta, tb, values, m):
    plt.figure(figsize=(8, 4))

    plt.subplot()
    plt.plot(ta, linestyle='--')
    plt.xlim((0, len(ta)))
    plt.plot(range(np.argmin(values), np.argmin(values) + m), ta[np.argmin(values):np.argmin(values) + m], c='g',
             label='Best Match')
    plt.legend(loc='best')
    plt.title('Time-Series')
    plt.ylim((-3, 3))

    plt.subplot()
    plt.plot(tb)
    plt.title('Query')
    plt.xlim((0, len(tb)))
    plt.ylim((-3, 3))

    plt.figure()
    plt.title('Matrix Profile')
    plt.plot(range(0, len(values)), values, '#ff5722')
    plt.plot(np.argmax(values), np.max(values), marker='x', c='r', ms=10)
    plt.plot(np.argmin(values), np.min(values), marker='^', c='g', ms=10)

    plt.xlim((0, len(ta)))
    plt.xlabel('Index')
    plt.ylabel('Value')

    plt.show()

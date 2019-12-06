*****************************************************************
*      Policy-agnostic Cache Size Inference (low load)          *
*****************************************************************

test_size_inferece.m: simulation driver; evaluating RCSE by varying #repetitions n, probing rate lm_a, initial guess c0

RCSE.m: implement policy-agnostic cache size inference based on 'forward-backward-probing'

forwardbackwardprobing.m: subroutine called by RCSE.m

***********************************************************
*   Size-aware Cache Policy Inference (low load)          *
***********************************************************

test_policy_inference.m: simulation driver; evaluating RCPD by varying #repetitions N, probing rate lm_a, estimated cache size C_est

RCPD.m: size-aware policy inference based on 'flush-promote-evict-test'

flushpromoteevicttest.m: subroutine called by RCPD.m

****************************************************************
*    characteristic-time-based policy inference (heavy load)   *
****************************************************************

test_characteristic_time.m: driver; evaluate accuracy of characteristic time estimation and accuracy of detecting policy by comparing the measured hit ratio with the TTL approximation based on the estimated characteristic time

characteristic_time_estimation.m: algorithm to estimate characteristic time (from Deghgan13MILCOM)

characteristic_time.m: compute the true characteristic time by solving the characteristic equation

characteristic_time_FIFO.m: (subroutine) called by characteristic_time.m

characteristic_time_LRU.m: (subroutine) called by characteristic_time.m

hit_ratio_FIFO/LRU.m: TTL approximation of the hit probability under FIFO or LRU

************************************************************************
*    policy-aware cache size & user parameter inference (heavy load)   *
************************************************************************


test_joint_inference_v2/v3.m: jointly inference F, lmd, C and alpha under FIFO or LRU on synthetic data, the difference is minor, like different parameters

test_joint_inference_trace.m: jointly inference F, lmd, C and alpha under FIFO or LRU on trace data

***********************************************************
*                DoS attack:  (heavy load)                *
***********************************************************

test_DoS.m: evaluate the theoretical/actual hit ratio for users under varying attack rate; separately compare equal rate vs. unequal rate, and under each of the policies: FIFO, LRU, LFU (theoretical prediction)

hit_ratio_with_attack_theory.m: prediction of hit ratio for users
hit_ratio_with_attack_sim.m: actual hit ratio for users from simulations

***********************************************************
*                Auxiliary functions                      *
***********************************************************

CacheAdd_withdelay.m: simulate a cache with a given size, a given policy ('FIFO' or 'LRU'), and a given insertion delay after misses dI

CacheAdd.m: (outdated) a cache with a given policy but no insertion delay (instantaneous insertion after each miss)

FifoAdd.m: (subroutine) FIFO cache (no insertion delay)

LruAdd.m: (subroutine) LRU cache (no insertion delay)

exp_trace.m: generate background traffic according to a Poisson process with Zipf popularity

poisson_trace.m: generate a Poisson point process (just timestamps, not the content IDs); for probing purposes

attack_trace.m: generate n Poisson processes with given rates (possibly unequal), each with a distinct content ID; for DoS attack 

NOTE: all the files named with wrong mean that estimated parameters are used for the attack or inference





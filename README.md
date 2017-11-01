GPUOthello2
====

GPU Accelerated endgame solver for Othello

## Usage

```
$ head probrem
1091780
!!%[nNNNNNnNNNNN
!!%zN{Nrn{w{Z{{{
!!&OrNXR{OoO{R{[
!!&WOZOXrZ[X{[{[
!!&WOZX[rZXx{z{[
!!&WOZxXxX{[{W{{
!!&wRzXroqxnRn{{
!!&zO{xRxQ{X{[o[
!!(RzOqQQRQQwNNN
$ ./solver
usage: ./solver INPUT OUTPUT DEPTH NAIVE_DEPTH
$ ./solver probrem result 10 9
n = 1091780
no error, elapsed: 6.443986s
```

### INPUT

Input file path

#### Input file format

```
n
board 1
board 2
...
board n
```

each board format is base81
for detail: https://github.com/primenumber/issen

### OUTPUT

Output file path

### DEPTH

Maximum search depth (without pass), equal to maximum number of empty position

### NAIVE\_DEPTH

Depth of naive search
Recommended: min(9, DEPTH-1)

## LICENSE

GPLv3

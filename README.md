GPUOthello2
====

GPU Accelerated endgame solver for Othello

## Usage

```
$ head problem
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
usage: ./solver INPUT OUTPUT DEPTH
$ ./solver probrem result 10
n = 1091780
no error, elapsed: 4.623645s, table update count: 0, table hit: 0, table find: 0
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

## LICENSE

GPLv3

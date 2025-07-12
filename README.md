### Compilating

#### C

```bash
gcc -fopenmp det.c -o det.exe
```

#### CUDA

```
nvcc det.cu -o det.exe -fmad=false
```

The flag -fmad=false is necessary for correct values at floating operations, without massive precision losses.

#### Executing

```bash
det.exe < input.txt
```

#### Output

```bash
Time (DET): 1.143000
820594634.419761
Time (TOT): 2.280000
```

where "Time (DET)" is the time in seconds of the parallelizable code and "Time (TOT)" is the total time of the algorithm, including the parallel part.

#### Simple test (992x992 matrix)

|            | 1 CPU  | 2 CPUs | 4 CPUs | 6 CPUs | 8 CPUs | 114,688 GPUs |
| ---------- | ------ | ------ | ------ | ------ | ------ | ------------ |
| Time (TOT) | 5.9606 | 3.6947 | 2.6355 | 2.6388 | 2.4805 | 1.2800       |
| Time (DET) | 4.8085 | 2.5352 | 1.4706 | 1.4691 | 1.3181 | 6.6104e-7    |

#### Test for 5000x5000 matrix

|            | 8 CPUs   | 114,688 GPUs |
| ---------- | -------- | ------------ |
| Time (TOT) | 191.0062 | 60.9828      |
| Time (DET) | 162.5468 | 3.5662e-5    |

# Initial State Optimization in Quantum Sensor Network

We model a quantum sensor network using techniques from quantum state discrimination. The interaction
between a qubit detector and the environment is described by a unitary operator, and we will assume that at
most one detector does interact. The task is to determine which one does or if none do. This involves choosing
an initial state of the detectors and a measurement. We consider global measurements in which all detectors are
measured simultaneously. We find that an entangled initial state can improve the detection probability, but this
advantage decreases as the number of detectors increases.

## Code
Example of running a Hill climbing program.

python main.py -us 2 -m Hill\ climbing -mi 100 -rn True -ns 3 -p 0.33333333333333 0.33333333333333 0.33333333333333 -em computational -ut 51 -ss 0 -od result/tmp -of varying_theta_3sensors_computational

For understanding the parameters: python main.py -h

## Paper 1

Physical Review A: https://caitaozhan.github.io/file/PhysRevA.QuantumSensor.pdf

Presentation: [YouTube](https://www.youtube.com/watch?v=c3u9HJypLog)

Welcome to cite :)
```
@article{PhysRevA.107.012435,
  title = {Discrete outcome quantum sensor networks},
  author = {Hillery, Mark and Gupta, Himanshu and Zhan, Caitao},
  journal = {Phys. Rev. A},
  volume = {107},
  issue = {1},
  pages = {012435},
  numpages = {6},
  year = {2023},
  month = {Jan},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevA.107.012435},
  url = {https://link.aps.org/doi/10.1103/PhysRevA.107.012435}
}
```

## Paper 2

https://arxiv.org/abs/2306.17401
```
@misc{zhan2023optimizing,
      title={Optimizing Initial State of Detector Sensors in Quantum Sensor Networks}, 
      author={Caitao Zhan and Himanshu Gupta and Mark Hillery},
      year={2023},
      eprint={2306.17401},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```

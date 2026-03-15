# Redshift-Distortion-Correction

Masters' thesis work related to correcting Redshift Distortions using AI.

AI architecture focuses on graph-based convolutions over a Delaunay tessellation graph,
where each node has **4 natural neighbors** and **5 learnable weights** (1 self + 4 neighbor).

The goal is to correct nonlinear regions consistently and agnostically for input data.

## Project Structure

```
Redshift-Distortion-Correction/
├── Project.toml                        — Julia package manifest & dependencies
├── src/
│   ├── RedshiftDistortionCorrection.jl — module entry point
│   ├── graph.jl                        — Delaunay tessellation → GNNGraph
│   ├── model.jl                        — NaturalNeighborConv layer & network
│   ├── data.jl                         — dataset loading & synthetic generation
│   ├── training.jl                     — training loop & checkpointing
│   └── utils.jl                        — metrics & helpers
├── test/
│   └── runtests.jl                     — test suite
└── examples/
    └── demo.jl                         — end-to-end usage example
```

## Key Dependencies

| Package | Purpose |
|---|---|
| [TetGen.jl](https://github.com/JuliaGeometry/TetGen.jl) | 3-D Delaunay tessellation |
| [Flux.jl](https://fluxml.ai/) | Deep-learning framework |
| [GraphNeuralNetworks.jl](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl) | GNN layers & message passing |
| [JLD2.jl](https://github.com/JuliaIO/JLD2.jl) | Model checkpointing |


## License

[GNU AGPL v3.0](LICENSE)

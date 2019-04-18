## 2D Ising Model simulator in Python3

Usage:

```python
import ising 

lattice = ising.IsingModel(40, 'r')
lattice.save_movie('ising.mp4')
lattice.runMovie(2.0, 1e5, -0.1, 1000)
lattice.finish_movie()
```

See source comments for more details and examples.

## License
The MIT License (MIT)

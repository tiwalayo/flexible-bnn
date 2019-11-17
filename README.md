# Tutorial for Bayesian neural networks vs Normal neural networks

This is a very simple two-notebook tutorial comparing a traditional neural network consisting of only 2 layers and a Bayesian neural network again with the same structure. The task is to classify the well-known iris dataset. This repository demonstrates the subtle differences and shows how to train/develop your own Bayesian neural network. 

I am also gradually adding other stuff related to probabilistic programming under the `pyro` subdirectory. Those are mainly tutorials that act as a resource and reference for my understanding and future implementation. 

Current examples: 

- [x] Monte Carlo Markov Chain and Mean Field VI 

## Getting Started

There are only two notebooks:

- `Bayesian neural network.ipynb` For the Bayesian neural networks
- `Normal neural network.ipynb` For the normal neural network

### Prerequisites

The requirements are in the `requirements.txt`. So simply: 

```
pip3 intall -r requirements.txt
```

In terms of literature on Bayesian neural networks, I strongly recommend Zoubin Ghahramani's NIPS talk: https://www.youtube.com/watch?v=v1BTHd5HXYE or his Nature review: https://www.repository.cam.ac.uk/bitstream/handle/1810/248538/Ghahramani%202015%20Nature.pdf


## Built With

* [PyTorch](https://pytorch.org/) - The deep learning library used
* [Pyro](http://docs.pyro.ai/en/0.2.1-release/index.html/) - The probabilistic deep learning library build on top of PyTorch

## Contributing

Any contributions are warmly welcome, I am planning on adding code also for:

- [ ] Convolutional neural network
- [ ] Regression


## Author

* **Martin Ferianc** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


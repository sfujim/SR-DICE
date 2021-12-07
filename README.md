# A Deep Reinforcement Learning Approach to Marginalized Importance Sampling with the Successor Representation

Code for Successor Representation DIstribution Correction Estimation (SR-DICE) a marginalized importance sampling method which builds off of deep successor representation. The paper will be presented at ICML 2021.

Code is provided for both continuous and discrete domains. Results were collected with [MuJoCo 1.50](http://www.mujoco.org/) on [OpenAI gym 0.17.2](https://github.com/openai/gym). Networks are trained using [PyTorch 1.4.0](https://github.com/pytorch/pytorch) and Python 3.7. 

### Usage

#### Continuous

Train expert:
```
python train_expert.py
```
Collect data & train SR-DICE:
```
python main.py
```

#### Discrete

Train expert:
```
python main.py --train_behavioral
```
Collect data:
```
python main.py --generate_buffer
```
Train SR-DICE:
```
python main.py
```

### Bibtex

```
@InProceedings{fujimoto2021srdice,
  title = 	 {A Deep Reinforcement Learning Approach to Marginalized Importance Sampling with the Successor Representation},
  author =       {Fujimoto, Scott and Meger, David and Precup, Doina},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {3518--3529},
  year = 	 {2021},
  volume = 	 {139},
  publisher =    {PMLR},
}
```

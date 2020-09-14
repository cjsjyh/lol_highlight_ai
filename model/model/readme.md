# Requirements 

Your model should include following methods:
```python
def train_wrapper(self, hps, dataset)
```
Function that calls `self.forward()` for training. Extract required data from dataset and call `forward()`. It should return `target` with result from `forward()`
```python
def eval_wrapper(self, hps, dataset)
```
Function that calls `self.forward()` for evaluation. Extract required data from dataset and call `forward()`. No need to return `target`.

```python
def forward(self, ...)
```
Pytorch function that trains/evaluates model.


# Sample
```python
class VASNet(nn.Module):

    def __init__(self):
        """ Custom Initialization """

    def train_wrapper(self, hps, dataset):
        seq = dataset['features'][...]
        target = dataset['gtscore'][...]
        seq_len = seq.shape[1]
        """ Custom code (get features from dataset) """
        return self.forward(seq,seq_len) + (target,)

    def eval_wrapper(self, hps, dataset):
        seq = dataset['features'][...]
        """ Custom code (get features from dataset) """
        return self.forward(seq, seq.shape[1])

    def forward(self, x, seq_len):
        """ Custom code """
        return y, att_weights_

```
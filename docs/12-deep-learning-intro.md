# Lesson 12 — Intro to Deep Learning (PyTorch)

In lesson 11 you wrote every gradient by hand. That works for two layers; it does not
scale to fifty. Deep learning frameworks exist to do exactly the bookkeeping you just did —
and nothing more mysterious than that. In this final lesson you rebuild lesson 11's MLP in
PyTorch, see the one-to-one correspondence between your NumPy lines and the framework
calls, and then swap in a small convolutional network that beats the MLP with 11× fewer
parameters. The dataset is Fashion-MNIST: same format as MNIST, harder problem.

## Concept

**From hand-written gradients to autograd.** In lesson 11, every line of the forward pass
had a matching hand-derived line in the backward pass. PyTorch's *autograd* (short for
"automatic differentiation") automates that matching. Under the hood, each tensor operation
records itself into a *computation graph* as it runs — a graph here is nothing more than a
memo of "this value came from adding/multiplying/etc. these other values", kept so torch can
retrace the steps later. The graph is built one operation at a time during the forward
pass, and then `loss.backward()` walks that graph in reverse, applying the chain rule —
exactly what your seven gradient lines did, generalized to any graph, of any depth,
automatically. Be clear about the division of labor: PyTorch automates the *derivative
bookkeeping*. It does not choose the architecture, the loss, the learning rate, the
initialization, or the data handling. Every knob you tuned in lesson 11 still exists; you
just stop differentiating by hand.

**Tensors.** A `torch.Tensor` is a NumPy array plus two abilities: it can track gradients,
and it can live on other devices (a GPU) without changing your code. If you know NumPy,
picture the same shaped grid of numbers as an `ndarray` — the two differences that matter
in this lesson are that operations on tensors get recorded for autograd (above), and every
parameter tensor carries a `.grad` slot that starts empty and fills in once you call
`loss.backward()`. `torch.from_numpy(...)` wraps an existing array without copying — the
scripts parse the IDX files with NumPy exactly as in lesson 11, then hand the arrays to
torch.

**Anatomy of a training loop.** Every PyTorch training loop — from this lesson to a large
language model — has the same five beats: forward (compute predictions), loss (score how
wrong they are), zero the gradients (clear old `.grad` values before computing new ones),
backward (fill `.grad` in via autograd), step (use `.grad` to move the weights). Side by
side with lesson 11:

| lesson 11 (NumPy, by hand)                       | lesson 12 (PyTorch)        |
|--------------------------------------------------|----------------------------|
| `z1 = x @ w1 + b1` … `p = softmax(z2)`           | `logits = model(x)`        |
| `-np.log(p[np.arange(m), y])`                    | `loss = loss_fn(logits, y)`|
| gradients recomputed fresh for each batch        | `optimizer.zero_grad()`    |
| the seven hand-written gradient lines            | `loss.backward()`          |
| `w1 -= LR * dw1` (× 4 parameters)                | `optimizer.step()`         |

Reading the table row by row reproduces exactly one pass over one mini-batch; the training
loop is just this table's right-hand column, repeated once per batch, for every epoch.

Two details worth internalizing: `nn.CrossEntropyLoss` takes *raw logits* — it fuses
log-softmax and the negative log-likelihood, which is the same softmax + cross-entropy
pairing (and the same clean $\hat{y} - y$ gradient) you derived in lesson 11, with the
numerical stability handled for you. And gradients in torch *accumulate*: each
`.backward()` call *adds* into the existing `.grad` buffers rather than overwriting them
(by design — it lets advanced training setups sum gradients over several mini-batches
before stepping, when one batch does not fit in memory). Forgetting
`optimizer.zero_grad()` therefore does not raise an error; it silently sums gradients
across batches and corrupts every update after the first.

**Why convolutions for images.** The MLP's first layer connects each of 784 pixels to each
of 128 hidden units: 100,352 weights, and a stroke detector learned in the top-left corner
must be relearned from zero for the center. A *convolution* instead slides one small
*filter* — also called a *kernel* (the code below names the argument `kernel_size`; "filter"
and "kernel" mean the same small weight matrix) — here 3×3, across the image:

- *Local patterns*: edges, corners and textures live in small neighborhoods; a 3×3 window
  is enough to detect them.
- *Weight sharing*: the same 9 weights are applied at every position, so a pattern learned
  anywhere is recognized everywhere — *translation equivariance* (shift the input and the
  feature map shifts with it: the same stroke detector fires wherever the stroke appears,
  with nothing relearned).
- *Few parameters*: a conv layer has $C_{\text{out}} \times (k^2 C_{\text{in}} + 1)$
  parameters. Here: conv1 $= 8 \times (9 \cdot 1 + 1) = 80$, conv2
  $= 16 \times (9 \cdot 8 + 1) = 1{,}168$, final linear $= 784 \times 10 + 10 = 7{,}850$ —
  **9,098 total, versus 101,770** for the MLP.

**Pooling.** Pooling is a fixed, unlearned down-sampling step, as opposed to a convolution's
learned filter. A 2×2 max-pool slides a 2×2 window over the feature map and keeps only the
strongest response in each block, halving each spatial dimension (28 → 14 → 7 after two
pools). Where weight sharing bought *equivariance* above (shift the input, the output
shifts to match), pooling buys a little translation *invariance* — the exact pixel position
of a feature stops mattering, only whether it fired somewhere inside the block — and
quarters the computation for every layer after it.

**Where to go next.** Almost everything past this lesson is scale and refinement of what
you have now seen: GPUs run the identical code faster (`model.to("cuda")` and move the
tensors along); deeper networks need help to train (dropout, batch normalization, weight
decay, residual connections); and attention-based *transformers* now dominate language and
much of vision. Two references cover the road ahead: Goodfellow, Bengio & Courville,
[*Deep Learning*](https://www.deeplearningbook.org/) (the theory), and
[*Dive into Deep Learning*](https://d2l.ai/) (runnable code, through transformers). The
habits from this repo — fix the seeds, print the metrics, compare against an expected
output — carry to models of any size.

## Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — 70,000 grayscale 28×28
images of clothing from Zalando Research, built as a drop-in replacement for MNIST (same
IDX format, same 60,000/10,000 split, same image size) but deliberately harder. License:
MIT.

The ten classes, indexed by label:

| label | class       | label | class      |
|-------|-------------|-------|------------|
| 0     | T-shirt/top | 5     | Sandal     |
| 1     | Trouser     | 6     | Shirt      |
| 2     | Pullover    | 7     | Sneaker    |
| 3     | Dress       | 8     | Bag        |
| 4     | Coat        | 9     | Ankle boot |

```powershell
.\scripts\download_data.ps1 fashion
```

Expected output:

```text
[get ] fashion-train-images-idx3-ubyte.gz
[ ok ] fashion-train-images-idx3-ubyte.gz (checksum verified)
[get ] fashion-train-labels-idx1-ubyte.gz
[ ok ] fashion-train-labels-idx1-ubyte.gz (checksum verified)
[get ] fashion-t10k-images-idx3-ubyte.gz
[ ok ] fashion-t10k-images-idx3-ubyte.gz (checksum verified)
[get ] fashion-t10k-labels-idx1-ubyte.gz
[ ok ] fashion-t10k-labels-idx1-ubyte.gz (checksum verified)
Done.
```

(If the files were downloaded earlier, you will see `[skip]` lines instead.)

As in lesson 11, both scripts train on the **first 10,000** training images and evaluate
on the **first 2,000** test images — a deterministic subset that keeps each run to seconds
on one CPU thread — and scale pixels to $[0, 1]$.

## From scratch

There is deliberately no `from_scratch.py` in this lesson:
[Lesson 11](11-neural-networks.md) **was** the from-scratch version — its
[`from_scratch.py`](../src/lesson11_neural_networks/from_scratch.py) is the hand-written
NumPy network that the scripts below rebuild. The MLP below is the same 784 → 128 (ReLU) →
10 network, and everything PyTorch does in one `loss.backward()` call is the seven gradient
lines you already wrote by hand. If any call in the scripts below feels magical, the cure
is to reread your own lesson 11 code — the "Step-by-step: reading the code" subsections
below point at exactly which lesson 11 line each PyTorch line replaces.

## With a library

### The MLP, in PyTorch

Read [`src/lesson12_deep_learning/with_library.py`](../src/lesson12_deep_learning/with_library.py)
side by side with lesson 11's script — the comments mark what replaced what: `nn.Linear`
owns the weight matrices you allocated by hand, a seeded `DataLoader` replaces
`rng.permutation`, `loss.backward()` replaces the backward pass, `optimizer.step()`
replaces the update lines.

#### Step-by-step: reading the code

Follow the file top to bottom. Each step below is doing exactly what a piece of lesson
11's `from_scratch.py` did — just written with torch instead of NumPy.

**1. Load the data — identical idea to lesson 11.** The IDX-parsing functions are the same
kind of code as lesson 11's `load_idx_images` / `load_idx_labels` (16-byte header, then raw
`uint8` pixels or labels). The one new step is handing the resulting NumPy arrays to torch:

```python
x_train_t = torch.from_numpy(x_train.astype(np.float32) / 255.0)
y_train_t = torch.from_numpy(y_train.astype(np.int64))
```

This is the same `/ 255.0` scaling lesson 11 does with `x_train.astype(np.float32) /
255.0`; `torch.from_numpy` just wraps the resulting array as a tensor, with no copy.

**2. Shuffle the data — `DataLoader` replaces `rng.permutation`.** Lesson 11 reshuffles by
hand every epoch (`order = rng.permutation(n)`) and then slices `BATCH`-sized chunks out of
`order` inside a manual loop. `with_library.py` gets the same effect from one object:

```python
loader = DataLoader(
    TensorDataset(x_train_t, y_train_t),
    batch_size=BATCH,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
    num_workers=0,
)
```

`shuffle=True` reshuffles every epoch, `batch_size=BATCH` does the chunking, and the
seeded `generator` plays the same role as lesson 11's seeded `rng`: run the script twice
and the batches come out in the same order both times.

**3. Define the model — `nn.Sequential` replaces `w1, b1, w2, b2`.** Lesson 11 allocates
four arrays by hand and writes the matrix products out explicitly (`z1 = x @ w1 + b1`, a
ReLU, then `z2 = a1 @ w2 + b2`). `with_library.py` declares the same shapes as a stack of
layers:

```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

`nn.Linear(784, 128)` owns a 784×128 weight matrix and a length-128 bias vector — exactly
`w1` and `b1` — and initializes them itself (a different default scheme than lesson 11's
He initialization, but the same idea: random, variance-scaled, never all zero).
`nn.ReLU()` is `np.maximum(z1, 0.0)`. The second `nn.Linear(128, 10)` is `w2` and `b2`.
There is no softmax layer here — it is folded into the loss, next.

**4. Define the loss — `nn.CrossEntropyLoss` replaces the softmax and `-np.log` lines.**
Lesson 11 computes softmax probabilities and then the negative log-likelihood by hand
(`e = np.exp(z2)`, `p = e / e.sum(...)`, `-np.log(p[np.arange(m), y])`).
`with_library.py` replaces both steps with one object, applied directly to the raw logits:

```python
loss_fn = nn.CrossEntropyLoss()
```

and inside the loop, `loss = loss_fn(logits, y)`. This is why the model above has no
softmax module: `nn.CrossEntropyLoss` expects raw logits and applies log-softmax
internally — numerically the same max-subtraction trick lesson 11 uses before
exponentiating.

**5. Define the optimizer — replaces the four `-= LR * d...` lines.** Lesson 11 updates
each of its four parameter arrays on its own line (`w1 -= LR * dw1`, and so on for `b1`,
`w2`, `b2`). `with_library.py` hands all of the model's parameters to one optimizer object,
once, outside the loop:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
```

`model.parameters()` yields the same four tensors (`nn.Linear`'s weight and bias, for both
layers) as one iterable; `optimizer.step()` (next) applies the identical update rule to
each of them.

**6. The training loop — `zero_grad` / `backward` / `step` replace the seven gradient
lines and the four update lines.**

```python
for x, y in loader:
    logits = model(x)          # forward pass (autograd records the graph)
    loss = loss_fn(logits, y)  # scalar batch loss
    optimizer.zero_grad()      # grads accumulate in torch — reset first
    loss.backward()            # replaces lesson 11's seven hand-written gradient lines
    optimizer.step()           # SGD update: p -= lr * p.grad for every parameter
    loss_sum += loss.item() * len(y)
```

Matched against lesson 11: `logits = model(x)` is the forward pass (`z1 = x @ w1 + b1`,
ReLU, `z2 = a1 @ w2 + b2`, all run inside `nn.Sequential`'s call). `loss = loss_fn(logits,
y)` is the cross-entropy computation above. `optimizer.zero_grad()` has no lesson-11
equivalent, because NumPy gradients were recomputed fresh every batch and never needed
clearing; torch's accumulate-by-default `.grad` buffers do. `loss.backward()` is the seven
lines from `dz2 = (p - y_onehot) / m` down to `db1 = dz1.sum(axis=0)`, run in reverse,
automatically. `optimizer.step()` is the four `-= LR * d...` lines, generalized to loop
over whatever parameters were registered.

```bash
python src/lesson12_deep_learning/with_library.py
```

Expected output:

```text
== Deep learning: PyTorch MLP ==
architecture: 784 -> 128 (ReLU) -> 10, 101770 parameters
train: first 10000 Fashion-MNIST train images, test: first 2000 t10k images
SGD: batch=64, lr=0.2, epochs=3

epoch 1  train_loss=0.9156  test_acc=0.7080
epoch 2  train_loss=0.5998  test_acc=0.6985
epoch 3  train_loss=0.5296  test_acc=0.7875

final test accuracy: 0.7875
```

Things to notice:

- **101,770 parameters** — the same count as lesson 11, because it is the same
  architecture. Only the plumbing changed. Verify it by hand: $(784 \times 128 + 128) +
  (128 \times 10 + 10) = 101{,}770$ — the same arithmetic as lesson 11's `w1.size + b1.size
  + w2.size + b2.size`.
- The same network that reached 0.9375 on MNIST digits gets 0.7875 here with an even
  shorter budget: telling a shirt from a pullover from a coat is genuinely harder than
  telling a 3 from an 8. That is exactly why Fashion-MNIST was created.
- Test accuracy dips at epoch 2 while train loss falls — the same SGD noise you saw in
  lesson 11. Three epochs on 10,000 images (471 updates) is a deliberately tiny budget.

### A small CNN

The second script, [`cnn.py`](../src/lesson12_deep_learning/cnn.py), keeps the data, the
loss, the optimizer, and the training loop **identical** and changes only the
architecture: two convolution + pooling blocks, then one linear layer.

#### Step-by-step: reading the code

`cnn.py` shares its data loading, loss, optimizer, and training loop with
`with_library.py` almost line for line. Read it as a diff against the MLP script, and
focus on what is actually different: the shape of the input and the model.

**1. One extra dimension: `unsqueeze(1)`.** The MLP fed each image to `nn.Linear` as a
flat 784-length vector. `nn.Conv2d` instead expects a 4-D tensor shaped `(batch, channels,
height, width)`. Fashion-MNIST images are grayscale — one channel — so the loader keeps
each image as a 28×28 grid and inserts a channel axis of size 1:

```python
x_train_t = torch.from_numpy(x_train.astype(np.float32) / 255.0).unsqueeze(1)
```

Everything else about loading the data — the IDX parsing, the `/ 255.0` scaling, the
`DataLoader` with its seeded generator — matches `with_library.py`. Even
`load_idx_images` is the same function apart from its last line: it returns shape `(n,
rows, cols)` here instead of the flattened `(n, rows*cols)` that `with_library.py` needs
for `nn.Linear`.

**2. The model — two conv+pool blocks, then one linear layer.**

```python
model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),   # (1, 28, 28) -> (8, 28, 28)
    nn.ReLU(),
    nn.MaxPool2d(2),                             # -> (8, 14, 14)
    nn.Conv2d(8, 16, kernel_size=3, padding=1),  # -> (16, 14, 14)
    nn.ReLU(),
    nn.MaxPool2d(2),                             # -> (16, 7, 7)
    nn.Flatten(),                                # -> 784
    nn.Linear(16 * 7 * 7, 10),                   # -> 10 logits
)
```

This is where the Concept section's parameter arithmetic comes from, layer by layer:

- `nn.Conv2d(1, 8, kernel_size=3, padding=1)` — 1 input channel, 8 output channels, one
  3×3 kernel per output channel: $C_{\text{out}} \times (k^2 C_{\text{in}} + 1) = 8 \times
  (9 \cdot 1 + 1) = 80$ parameters. `padding=1` pads the image by one pixel on each side so
  the 3×3 kernel produces an output the same size as its input (28×28 in, 28×28 out) —
  without it, the spatial size would shrink at every convolution instead of only at the
  pooling layers.
- `nn.MaxPool2d(2)` halves height and width (28 → 14) and **has no parameters at all**: it
  is a fixed rule (keep the max of each 2×2 block), not something learned.
- `nn.Conv2d(8, 16, kernel_size=3, padding=1)` — now 8 input channels (the previous
  layer's 8 output feature maps) feed 16 output channels: $16 \times (9 \cdot 8 + 1) =
  1{,}168$ parameters. This is the biggest jump in parameter count, because each of the 16
  output filters needs its own 3×3 kernel *for every one of the 8 input channels*.
- The second `nn.MaxPool2d(2)` takes 14 → 7.
- `nn.Flatten()` reshapes the `(16, 7, 7)` feature map into one length-784 vector
  ($16 \times 7 \times 7 = 784$ — coincidentally the same 784 as the original flattened
  image, though these 784 numbers are learned features, not raw pixels).
- `nn.Linear(16 * 7 * 7, 10)` is an ordinary fully-connected layer, exactly like the MLP's
  output layer: $784 \times 10 + 10 = 7{,}850$ parameters.

Add them up — $80 + 1{,}168 + 7{,}850 = 9{,}098$ — and that is exactly the `parameters:
CNN 9098` printed below, and the count quoted in the Concept section.

**3. Loss, optimizer, training loop — unchanged from `with_library.py`.**

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
```

and:

```python
for x, y in loader:  # identical loop to with_library.py
    logits = model(x)
    loss = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_sum += loss.item() * len(y)
```

Everything said about `with_library.py`'s training loop — what `zero_grad` / `backward` /
`step` each replace from lesson 11 — applies here unchanged. `loss.backward()` still walks
the computation graph in reverse; it does not need special-case code for convolutions,
because `nn.Conv2d` is built out of ordinary tensor operations autograd already knows how
to differentiate. The only numeric difference in this block is `LR = 0.25` instead of
`0.2` — a tuning choice, not a change to the training logic.

```bash
python src/lesson12_deep_learning/cnn.py
```

Expected output:

```text
== Deep learning: PyTorch CNN ==
architecture: conv 1->8 (3x3) + pool -> conv 8->16 (3x3) + pool -> linear 784 -> 10
parameters: CNN 9098 vs MLP 101770
train: first 10000 Fashion-MNIST train images, test: first 2000 t10k images
SGD: batch=64, lr=0.25, epochs=3

epoch 1  train_loss=1.3551  test_acc=0.5910
epoch 2  train_loss=0.6491  test_acc=0.7075
epoch 3  train_loss=0.5290  test_acc=0.8095

final test accuracy: 0.8095
```

Things to notice:

- **The CNN wins: 0.8095 vs 0.7875 — with 11× fewer parameters** (9,098 vs 101,770). More
  parameters is not more accuracy; an architecture that matches the structure of the data
  (local patterns, position-independence) is worth more than raw capacity. See the
  step-by-step breakdown above for exactly where each of those 9,098 parameters comes from.
- The CNN starts *worse* (0.5910 after epoch 1): it is deeper, so early gradients are less
  direct. It overtakes the MLP by the third epoch and its curve is still climbing — given
  the full 60,000 training images, the same architecture climbs into the high 0.80s within
  a few epochs.
- Both runs are bit-for-bit reproducible: seeded weight init (`torch.manual_seed`), a
  seeded `DataLoader` shuffle, and a single CPU thread (`torch.set_num_threads(1)`) so
  floating-point sums always happen in the same order.

## Check your understanding

1. Delete `optimizer.zero_grad()` from the loop. Which quantity becomes silently wrong,
   and what would you expect training to do?
2. Compute conv2's parameter count (1,168) by hand from its shape: 16 output channels,
   8 input channels, 3×3 kernels, plus biases.
3. The CNN beats the MLP with 11× fewer parameters. What assumption about images does it
   exploit — and name a kind of data from an earlier lesson where that assumption fails,
   making a convolution the wrong choice.
4. `nn.CrossEntropyLoss` expects raw logits. What goes wrong if you add
   `nn.Softmax(dim=1)` as the last layer of the model anyway?
5. Which lines change to train on a GPU — and why might the printed numbers then no
   longer match this doc digit-for-digit, even with the same seeds?
6. If you removed `padding=1` from both `nn.Conv2d` calls in `cnn.py`, each convolution
   would shrink the spatial size instead of preserving it (a 3×3 kernel with no padding
   turns a 28×28 map into a 26×26 one). Trace the sizes through both conv+pool blocks.
   Which layer's parameter count would have to change to match, and which two layers'
   parameter counts would stay exactly the same?
7. Both scripts build their `DataLoader` with `generator=torch.Generator().manual_seed(SEED)`,
   using the same `SEED` already passed to `torch.manual_seed`. Why use a second, separate
   seeded generator for the shuffle instead of relying on the one `torch.manual_seed`
   already set up — and what would become fragile about reproducing this doc's numbers if
   the `generator=` argument were dropped?

From here, you've reached the end of the ML Journey — congratulations! If you want to keep
going, revisit a lesson's "Check your understanding" questions, or propose a new lesson
(see [CONTRIBUTING.md](../CONTRIBUTING.md)).

Before you move on, however, add an entry in the "Reproduction Log" at the bottom of this
page, following the same format: use `yyyy-mm-dd`, make sure you're using a commit id
that's on the master branch of `ml-foundations`, and use its 7-hexadecimal prefix for the
link anchor text. In the description of your pull request, please provide some details on
your setup (e.g., operating system, environment and configuration, etc.). In addition,
also provide some indication of success (e.g., everything worked) or document issues you
encountered. If you think this guide can be improved in any way (e.g., you caught a typo
or think a clarification is warranted), feel free to include it in the pull request.

## Reproduction Log

<!-- Append one line per reproduction below this comment. Keep chronological order. -->

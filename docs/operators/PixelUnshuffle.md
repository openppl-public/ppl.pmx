# PixelUnshuffle

Reverse the PixelShuffle operation.

Reverses the PixelShuffle operation by rearranging elements in a tensor of shape $\left(*, H \times r, W \times r, C\right)$  to a tensor of shape $\left(*, H, W, C \times r^2 \right)$, where r is a scale factor.

## Attributes/Parameters

### `scale_factor`: int

factor to decrease spatial resolution by.

### `data_layout`: str

input data layout, only support 'nhwc'

## Inputs

### `X`: tensor(T)

Shape: $\left(*, H \times r, W \times r, C\right)$

## Outputs

### `Y`: tensor(T)

Shape: $\left(*, H, W, C \times r^2 \right)$
## Type Constraints

### `T`: float32, float16

import torch
from typing import List


def pseudo_quantize_linear_weight(w, n_bit=4, zero_point=True, group_size=-1):
    org_w_shape = w.shape #(out_features, in_features)
    if group_size > 0:
        assert org_w_shape[-1] % group_size == 0
        w = w.reshape(-1, group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        w = (torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros) * scales
    else:
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = - 2 ** (n_bit - 1)
        scales = max_val / max_int
        zeros = None
        w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

    w = w.reshape(org_w_shape)
    scales = scales.view(w.shape[0], -1)
    zeros = zeros.view(w.shape[0], -1) if zeros is not None else zeros

    return w, scales, zeros

class Int4QuantUtils():

    @staticmethod
    def pack(imatrix: torch.Tensor, storage_bits: int=16,
             q_bits: int=4, direction: str = "row"):
        """
        Packs a 4-bit integer matrix into a packed 16/32 bit integer matrix.
        Args:
            imatrix (torch.Tensor): matrix of integers
            storage_bits (int): number of bits to storage qmatrix
            q_bits (int): quantize bits
            direction (str): direction of packing, either "column" or "row"

        Returns:
            qmatrix (torch.Tensor): packed matrix of integers
        """
        shifts = torch.arange(0, storage_bits, q_bits, device=imatrix.device)

        imatrix = imatrix.to(torch.int8) & 0x0F
        pack_num = storage_bits // q_bits

        if direction == "column":
            imatrix = imatrix.view(-1, imatrix.shape[1] // pack_num, pack_num)
            qmatrix = torch.bitwise_left_shift(imatrix, shifts[None, None, :]).sum(dim=-1)

        elif direction == "row":
            imatrix = imatrix.view(imatrix.shape[0] // pack_num, pack_num, -1)
            qmatrix = torch.bitwise_left_shift(imatrix, shifts[None, :, None]).sum(dim=1)

        qmatrix = qmatrix.to(torch.int16) if storage_bits == 16  else qmatrix.to(torch.int32)

        return qmatrix

    @staticmethod
    def unpack(qmatrix: torch.Tensor, storage_bits: int=16,
               q_bits: int=4, direction: str = "row"):
        """
        Unpacks a 16/32 bit packed integer matrix into a 4-bit integer matrix.

        Args:
            qmatrix (torch.Tensor): matrix of packed integers
            storage_bits (int): number of bits to storage qmatrix
            q_bits (int): quantize bits
            direction (str): direction of unpacking, either "column" or "row"

        Returns:
            imatrix (torch.Tensor): matrix of integers
        """
        shifts = torch.arange(0, storage_bits, q_bits, device=qmatrix.device)

        if direction == "column":
            imatrix = torch.bitwise_right_shift(
                qmatrix[:, :, None], shifts[None, None, :]
            ).view(qmatrix.shape[0], -1)

        elif direction == "row":
            imatrix = torch.bitwise_right_shift(
                qmatrix[:, None, :], shifts[None, :, None]
            ).view(-1, qmatrix.shape[-1])

        imatrix = imatrix.to(torch.int8) & 0x0F  # eventually correct overflow

        return imatrix


    @staticmethod
    def quantize_fp16_to_int4(fmatrix: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
                              group_size: int=128, n_bits: int=4):
        """
        Quantizes a matrix of 16-bit floats into a matrix of unsigned 4 bit integers.

        Args:
            fmatrix (torch.Tensor): matrix of 16-bit floats
            scales (torch.Tensor): matrix of 16-bit floats
            zeros (torch.Tensor): matrix of 8 bit integers
            group_size (int): group size

        Returns:
            imatrix (torch.Tensor): matrix of unsigned 4 bit integers
        """

        if zeros is None or zeros.nelement() == 0:
            offset = 2 ** (n_bits - 1)
        else:
            zeros = zeros.to(torch.int8) & 0x0F
            offset = zeros.repeat_interleave(group_size, dim=1)

        imatrix = torch.round(
            fmatrix / scales.repeat_interleave(group_size, dim=1) + offset)

        imatrix = imatrix.to(torch.uint8) & 0x0F

        return imatrix


    @staticmethod
    def dequantize_int4_to_fp16(imatrix: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
                                group_size: int=128, n_bits: int=4):
        """
        Dequantizes a unsigned 4/8 bit integer matrix into a float matrix.

        Args:
            imatrix (torch.Tensor): matrix of unsigned 4 bit integers
            scales (torch.Tensor): matrix of 16-bit floats
            zeros (torch.Tensor): matrix of 4-bit integers
            group_size (int): group size

        Returns:
            fmatrix (torch.Tensor): matrix of 16-bit floats
        """
        if zeros is None or zeros.nelement() == 0:
            offset = 2 ** (n_bits - 1)
        else:
            zeros = zeros.to(torch.int8) & 0x0F
            offset = zeros.repeat_interleave(group_size, dim=1)
        imatrix = imatrix.to(torch.int8) & 0x0F
        fmatrix = (imatrix - offset) * scales.repeat_interleave(group_size, dim=1)
        fmatrix = fmatrix.to(torch.float16)

        return fmatrix



if __name__ == "__main__":
    layer = torch.nn.Linear(in_features=512, out_features=2048, dtype=torch.float16)
    weight = layer.weight #shape -> (output_features, in_features)
    qdq_w, scale, zp = pseudo_quantize_linear_weight(weight, n_bit=4, zero_point=False, group_size=128)
    int4_w = Int4QuantUtils.quantize_fp16_to_int4(qdq_w, scale, zp, 128)
    packed_int32_w = Int4QuantUtils.pack(int4_w)
    unpacked_int4_w = Int4QuantUtils.unpack(packed_int32_w)
    qdq_w_after_pack = Int4QuantUtils.dequantize_int4_to_fp16(unpacked_int4_w, scale, zp, 128)
    print ('diff after pack and unpack: \n', qdq_w - qdq_w_after_pack)

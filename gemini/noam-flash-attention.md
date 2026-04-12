Here is a canonical, highly readable implementation of **Flash Attention** (specifically the forward pass of Flash Attention 2\) utilizing Noam Shazeer’s shape-suffix convention.

Flash Attention is notoriously dense and difficult to read because it relies heavily on block-tiling and computing an "online softmax" trick to avoid materializing the massive attention matrix. However, by explicitly adding row (R) and column (C) block sizes to our dimension key and using torch.einsum, the notoriously complex block algebra becomes beautifully straightforward.

*(Note: I silently fixed a few minor variable name typos from Noam's original pseudo-code snippet, such as hiddden\_BLD, the missing layer\_num iterator, and projecting values using w\_v\_DHK to make the snippet cohesive).*

Python

""" Example Flash Attention code with shape suffixes.

This code is incomplete and possibly has bugs. Don't try to run it.  
Its purpose is to illustrate shape suffixes for complex block operations.

Dimension key:

B: batch size  
L: sequence length  
M: memory length (length of sequence being attended to)  
D: model dimension (sometimes called d\_model or embedding\_dim)  
V: vocabulary size  
F: feed-forward subnetwork hidden size  
H: number of attention heads in a layer  
K: size of each attention key or value (sometimes called d\_kv)  
R: row block size (query sequence length in a single tile)  
C: column block size (memory sequence length in a single tile)  
"""

import torch

def transformer(input\_token\_id\_BL, params):  
    hidden\_BLD \= params.embedding\_VD\[input\_token\_id\_BL\]  
    for layer\_num in range(params.num\_layers):  
        hidden\_BLD \+= flash\_attention(hidden\_BLD, params.attention\_params\[layer\_num\])  
        hidden\_BLD \+= ffn(hidden\_BLD, params.ffn\_params\[layer\_num\])  
    hidden\_BLD \= layer\_norm(hidden\_BLD, params.final\_layernorm\_params)  
    logits\_BLV \= torch.matmul(hidden\_BLD, params.embedding\_VD.T)  
    return logits\_BLV

def ffn(input\_BLD, params):  
    input\_BLD \= layer\_norm(input\_BLD, params.layernorm\_params)  
    hidden\_BLF \= torch.gelu(torch.matmul(input\_BLD, params.w\_in\_DF))  
    output\_BLD \= torch.matmul(hidden\_BLF, params.w\_out\_FD)  
    return output\_BLD

def flash\_attention(input\_BLD, params, Br=64, Bc=64, is\_causal=True):  
    input\_BLD \= layer\_norm(input\_BLD, params.layernorm\_params)  
      
    \# We use 'M' in the einsum string to represent memory length (M \== L for self-attention)  
    query\_BLHK \= torch.einsum('BLD,DHK-\>BLHK', input\_BLD, params.w\_q\_DHK)  
    key\_BMHK   \= torch.einsum('BMD,DHK-\>BMHK', input\_BLD, params.w\_k\_DHK)  
    value\_BMHK \= torch.einsum('BMD,DHK-\>BMHK', input\_BLD, params.w\_v\_DHK)  
      
    B, L, H, K \= query\_BLHK.shape  
    \_, M, \_, \_ \= key\_BMHK.shape  
      
    \# Initialize output in global High Bandwidth Memory (HBM)  
    out\_BLHK \= torch.zeros\_like(query\_BLHK)  
      
    \# 1\. Outer loop over sequence blocks (Rows of the attention matrix)  
    for i in range(0, L, Br):  
        query\_BRHK \= query\_BLHK\[:, i:i+Br, :, :\]  
        R \= query\_BRHK.shape\[1\] \# Actual row block size (handles sequence boundaries safely)  
          
        \# Initialize running statistics for the online softmax trick (in fast SRAM)  
        max\_BHR \= torch.full((B, H, R), float('-inf'), device=input\_BLD.device)  
        sum\_BHR \= torch.zeros((B, H, R), device=input\_BLD.device)  
        out\_BRHK \= torch.zeros\_like(query\_BRHK)  
          
        \# 2\. Inner loop over memory blocks (Columns of the attention matrix)  
        for j in range(0, M, Bc):  
            \# Causal mask optimization: completely skip fully masked future blocks  
            if is\_causal and j \>= i \+ R:  
                break  
                  
            key\_BCHK   \= key\_BMHK\[:, j:j+Bc, :, :\]  
            value\_BCHK \= value\_BMHK\[:, j:j+Bc, :, :\]  
            C \= key\_BCHK.shape\[1\]  
              
            \# Unnormalized attention scores for this block pair  
            logits\_BHRC \= torch.einsum('BRHK,BCHK-\>BHRC', query\_BRHK, key\_BCHK)  
            logits\_BHRC /= K \*\* 0.5  
              
            \# Causal mask trick applied dynamically within the block  
            if is\_causal:  
                \# i+r is absolute query index, j+c is absolute key index  
                q\_idx\_R1 \= torch.arange(i, i+R, device=input\_BLD.device).unsqueeze(1)  
                k\_idx\_1C \= torch.arange(j, j+C, device=input\_BLD.device).unsqueeze(0)  
                masked\_out\_RC \= q\_idx\_R1 \< k\_idx\_1C  
                logits\_BHRC \+= torch.where(masked\_out\_RC, float('-inf'), 0.0)  
                  
            \# \--- Flash Attention Softmax Trick \---  
              
            \# Find the new maximum for numerical stability  
            block\_max\_BHR \= torch.max(logits\_BHRC, dim=-1).values  
            new\_max\_BHR \= torch.maximum(max\_BHR, block\_max\_BHR)  
              
            \# Shift logits to prevent overflow. (nan\_to\_num catches fully masked \-inf rows)  
            shifted\_logits\_BHRC \= logits\_BHRC \- new\_max\_BHR.unsqueeze(-1)  
            shifted\_logits\_BHRC \= torch.nan\_to\_num(shifted\_logits\_BHRC, nan=float('-inf'))  
            exp\_logits\_BHRC \= torch.exp(shifted\_logits\_BHRC)  
              
            block\_sum\_BHR \= torch.sum(exp\_logits\_BHRC, dim=-1)  
              
            \# Calculate correction factor to decay old stats based on the new max  
            scale\_BHR \= torch.exp(max\_BHR \- new\_max\_BHR)  
            scale\_BHR \= torch.nan\_to\_num(scale\_BHR, nan=0.0)  
              
            \# Update the running denominator sum  
            sum\_BHR \= sum\_BHR \* scale\_BHR \+ block\_sum\_BHR  
              
            \# Calculate the new weighted values for this block  
            wtd\_values\_BRHK \= torch.einsum('BHRC,BCHK-\>BRHK', exp\_logits\_BHRC, value\_BCHK)  
              
            \# Update the running output   
            out\_BRHK \= torch.einsum('BRHK,BHR-\>BRHK', out\_BRHK, scale\_BHR) \+ wtd\_values\_BRHK  
              
            \# Commit loop variables for the next memory block iteration  
            max\_BHR \= new\_max\_BHR  
              
        \# 3\. Final normalization step (divide by the final running sum)  
        out\_BRHK \= torch.einsum('BRHK,BHR-\>BRHK', out\_BRHK, 1.0 / sum\_BHR)  
          
        \# Write updated block back to global HBM tensor  
        out\_BLHK\[:, i:i+R, :, :\] \= out\_BRHK

    \# 4\. Final dense projection  
    out\_BLD \= torch.einsum('BLHK,HKD-\>BLD', out\_BLHK, params.w\_o\_HKD)  
    return out\_BLD

### **Why shape suffixes make Flash Attention remarkably readable:**

1. **Transposes Disappear:** Traditional PyTorch attention logic requires deeply nested transformations like query.transpose(1, 2\) @ key.transpose(1, 2).transpose(2, 3\) just to align axes for block multiplication. Suffixes make torch.einsum('BRHK,BCHK-\>BHRC', ...) profoundly trivial.  
2. **Implicit Matrix Broadcasting:** Decaying the accumulated outputs by the max-scaling factor traditionally demands visual horrors like out \* scale.unsqueeze(1).unsqueeze(-1). Using shape suffixes, torch.einsum('BRHK,BHR-\>BRHK', out, scale) tells einsum to do the heavy lifting of flawlessly matching the B, H, and R dimensions for you.  
3. **Safe Reductions:** Knowing your unnormalized block logic sits in logits\_BHRC guarantees that calling .max(dim=-1) and .sum(dim=-1) collapses the C suffix. Visually, you instantly know it cleanly resolves into our target running state \_BHR tracking tensors without a second thought.
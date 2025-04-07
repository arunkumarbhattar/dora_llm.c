/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
*/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#ifdef OMP
#include <omp.h>
#endif
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#include "llmc/dataloader.h"

typedef struct {
    float* magnitude; // magnitude vector (trainable)
    float* A;         // low-rank matrix A
    float* B;         // low-rank matrix B
    float alpha;      // scaling factor
    int rank;         // rank of the low-rank matrices
    int in_dim;       // input dimension
    int out_dim;      // output dimension
} DoRALayer;

DoRALayer construct_dora_layer(float* magnitude, float* A, float* B,
                               int in_dim, int out_dim, int rank, float alpha) {
    DoRALayer layer;
    layer.magnitude = magnitude;
    layer.A = A;
    layer.B = B;
    layer.in_dim = in_dim;
    layer.out_dim = out_dim;
    layer.rank = rank;
    layer.alpha = alpha;
    return layer;
}

void dora_init(DoRALayer* layer, int in_dim, int out_dim, int rank, float alpha, float* pretrained_weight) {
    // Initialize magnitude with L2 norm + epsilon
    for (int i = 0; i < out_dim; i++) {
        float sum_sq = 1e-6f;  // Start with epsilon
        for (int j = 0; j < in_dim; j++) {
            float val = pretrained_weight[i * in_dim + j];
            sum_sq += val * val;
        }
        layer->magnitude[i] = sqrtf(sum_sq);
    }

    // He initialization for low-rank matrices
    float std_dev = sqrtf(2.0f / (in_dim + rank));
    for (int i = 0; i < in_dim * rank; i++) {
        layer->A[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * std_dev;
    }

    std_dev = sqrtf(2.0f / (rank + out_dim));
    for (int i = 0; i < rank * out_dim; i++) {
        layer->B[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * std_dev;
    }

    // Scale alpha by 1/sqrt(rank)
    layer->alpha = alpha / sqrtf(rank);
}

void dora_forward(float* output, float* input, DoRALayer* layer, float* pretrained_weight) {
    // Check that the DoRALayer is properly initialized.
    if (!layer || !layer->magnitude || !layer->A || !layer->B || !pretrained_weight) {
        fprintf(stderr, "dora_forward: Invalid pointers passed to function.\n");
        exit(EXIT_FAILURE);
    }

    int in_dim = layer->in_dim;
    int out_dim = layer->out_dim;
    int rank = layer->rank;

    // Allocate memory for the directional component.
    float* direction = (float*)mallocCheck(in_dim * out_dim * sizeof(float));

    // Compute directional component: each row of pretrained_weight is divided by the corresponding magnitude.
    for (int i = 0; i < out_dim; i++) {
        float mag = layer->magnitude[i];
        for (int j = 0; j < in_dim; j++) {
            int idx = i * in_dim + j;
            if (mag > 1e-6f)
                direction[idx] = pretrained_weight[idx] / mag;
            else
                direction[idx] = 0.0f;
        }
    }

    // Allocate temporary storage for the low-rank contribution.
    float* temp = (float*)mallocCheck(rank * sizeof(float));

    // Compute the output:
    //   output = input dot (direction + alpha * (A * B)), then multiplied by magnitude.
    for (int b = 0; b < out_dim; b++) {
        float result = 0.0f;

        // Compute temp[r] = (A^T * input)[r]
        for (int r = 0; r < rank; r++) {
            temp[r] = 0.0f;
            for (int a = 0; a < in_dim; a++) {
                temp[r] += input[a] * layer->A[a * rank + r];
            }
        }

        // Add contribution from the normalized directional weights.
        for (int a = 0; a < in_dim; a++) {
            result += input[a] * direction[b * in_dim + a];
        }

        // Add the low-rank update: alpha * (A^T * input)[r] * corresponding B.
        for (int r = 0; r < rank; r++) {
            result += layer->alpha * temp[r] * layer->B[r * out_dim + b];
        }

        // Multiply by the original magnitude.
        output[b] = result * layer->magnitude[b];
    }

    free(temp);
    free(direction);
}

void dora_backward(float* dinput, float* dmagnitude, float* dA, float* dB,
                   float* doutput, float* input, DoRALayer* layer, float* pretrained_weight) {
    // Validate all required pointers
    if (!layer || !layer->magnitude || !layer->A || !layer->B || !pretrained_weight) {
        fprintf(stderr, "dora_backward: Invalid pointers passed to function.\n");
        exit(EXIT_FAILURE);
    }

    const int in_dim = layer->in_dim;
    const int out_dim = layer->out_dim;
    const int rank = layer->rank;
    const float alpha = layer->alpha;

    // Zero-initialize gradient buffers
    memset(dmagnitude, 0, out_dim * sizeof(float));
    memset(dA, 0, in_dim * rank * sizeof(float));
    memset(dB, 0, rank * out_dim * sizeof(float));

    // Backprop through magnitude scaling
    for (int o = 0; o < out_dim; o++) {
        float mag = layer->magnitude[o];
        if (mag < 1e-6f) mag = 1e-6f;  // Numerical stability

        // Compute direction vector
        float* direction = (float*)mallocCheck(in_dim * sizeof(float));
        for (int i = 0; i < in_dim; i++) {
            direction[i] = pretrained_weight[o * in_dim + i] / mag;
        }

        // Magnitude gradient
        dmagnitude[o] = 0.0f;
        for (int i = 0; i < in_dim; i++) {
            dmagnitude[o] += doutput[o] * direction[i] * input[i];
        }

        // Low-rank adaptation gradients
        for (int r = 0; r < rank; r++) {
            float a_grad = 0.0f, b_grad = 0.0f;
            for (int i = 0; i < in_dim; i++) {
                a_grad += input[i] * layer->B[r * out_dim + o];
            }
            for (int i = 0; i < in_dim; i++) {
                b_grad += layer->A[i * rank + r] * input[i];
            }
            dA[r] += alpha * a_grad * doutput[o];
            dB[r * out_dim + o] += alpha * b_grad * doutput[o];
        }

        free(direction);
    }
}

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

void encoder_forward(float* out,
                   int* inp, float* wte, float* wpe,
                   int B, int T, int C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            int ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            float* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            float* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

void encoder_backward(float* dwte, float* dwpe,
                      float* dout, int* inp,
                      int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* dwte_ix = dwte + ix * C;
            float* dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                float d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            float mean_bt = mean[b * T + t];
            float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

void matmul_forward_naive(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int B, int T, int C, int OC) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int bt = b * T + t;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++) {
                    val += inp[bt * C + i] * weight[o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

void matmul_forward(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct or fallback to naive version
    const int LOOP_UNROLL = 8;
    if (B*T % LOOP_UNROLL != 0) {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        return;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #pragma omp parallel for
    for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        for (int o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            float result[LOOP_UNROLL];
            // initialize the bias, if it exists
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            for (int i = 0; i < C; i++) {
                float w = weight[i + o * C];
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    int bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                int bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

void matmul_backward(float* dinp, float* dweight, float* dbias,
                     const float* dout, const float* inp, const float* weight,
                     int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                const float* wrow = weight + o*C;
                float d = dout_bt[o];
                for (int i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel for
    for (int o = 0; o < OC; o++) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                const float* dout_bt = dout + b * T * OC + t * OC;
                const float* inp_bt = inp + b * T * C + t * C;
                float* dwrow = dweight + o*C;
                float d = dout_bt[o];
                if (dbias != NULL) { dbias[o] += d; }
                for (int i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

void attention_forward(float* out, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

void attention_backward(float* dinp, float* dpreatt, float* datt,
                        float* dout, float* inp, float* att,
                        int B, int T, int C, int NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.f / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                float* dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float* out, float* inp, int N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#pragma float_control(precise, on, push)
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
void gelu_backward(float* dinp, float* inp, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#pragma float_control(pop)

void residual_forward(float* out, float* inp1, float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

void residual_backward(float* dinp1, float* dinp2, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

void softmax_forward(float* probs, float* logits, int B, int T, int V, int Vp) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // probs <- softmax(logits)
            float* logits_bt = logits + b * T * Vp + t * Vp;
            float* probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = -10000.0f; // TODO something better
            for (int i = 0; i < V; i++) {
                if (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            for (int i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for (int i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

void crossentropy_forward(float* losses,
                          float* probs, int* targets,
                          int B, int T, int Vp) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // loss = -log(probs[target])
            float* probs_bt = probs + b * T * Vp + t * Vp;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

void crossentropy_softmax_backward(float* dlogits,
                           float* dlosses, float* probs, int* targets,
                           int B, int T, int V, int Vp) {
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            float* probs_bt = probs + b * T * Vp + t * Vp;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
    int use_dora;       // Flag to use DoRA instead of regular fine-tuning
    int dora_rank;      // Rank for DoRA adaptation
    float dora_alpha;   // Scaling factor for DoRA
} GPT2Config;

// the parameters of the model
#define NUM_PARAMETER_TENSORS 28
typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C)
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    float* fcw; // (L, 4*C, C)
    float* fcb; // (L, 4*C)
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)
    float* lnfw; // (C)
    float* lnfb; // (C)

    // DoRA specific parameters
    float* qkv_magnitude;
    float* qkv_A;
    float* qkv_B;
    float* attproj_magnitude;
    float* attproj_A;
    float* attproj_B;
    float* fc_magnitude;
    float* fc_A;
    float* fc_B;
    float* fcproj_magnitude;
    float* fcproj_A;
    float* fcproj_B;
} ParameterTensors;

// This function computes the sizes of the base (pretrained) parameters.
// Since our checkpoint file does not include DoRA parameters, we set the sizes
// for indices 16..27 to zero. Later, we allocate and initialize DoRA parameters separately.
void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    // Base model parameters (these are read from the checkpoint)
    param_sizes[0] = Vp * C;           // wte
    param_sizes[1] = maxT * C;           // wpe
    param_sizes[2] = L * C;              // ln1w
    param_sizes[3] = L * C;              // ln1b
    param_sizes[4] = L * (3 * C) * C;      // qkvw
    param_sizes[5] = L * (3 * C);          // qkvb
    param_sizes[6] = L * C * C;            // attprojw
    param_sizes[7] = L * C;                // attprojb
    param_sizes[8] = L * C;                // ln2w
    param_sizes[9] = L * C;                // ln2b
    param_sizes[10] = L * (4 * C) * C;       // fcw
    param_sizes[11] = L * (4 * C);           // fcb
    param_sizes[12] = L * C * (4 * C);       // fcprojw
    param_sizes[13] = L * C;               // fcprojb
    param_sizes[14] = C;                   // lnfw
    param_sizes[15] = C;                   // lnfb

    if (config.use_dora) {
        size_t L = config.num_layers;
        size_t rank = config.dora_rank;
        size_t C = config.channels;

        // DoRA parameter sizes
        param_sizes[16] = L * 3*C;        // qkv_magnitude
        param_sizes[17] = L * C*rank;     // qkv_A
        param_sizes[18] = L * rank*3*C;   // qkv_B
        param_sizes[19] = L * C;          // attproj_magnitude
        param_sizes[20] = L * C*rank;     // attproj_A
        param_sizes[21] = L * rank*C;     // attproj_B
        param_sizes[22] = L * 4*C;        // fc_magnitude
        param_sizes[23] = L * C*rank;     // fc_A
        param_sizes[24] = L * rank*4*C;   // fc_B
        param_sizes[25] = L * C;          // fcproj_magnitude
        param_sizes[26] = L * 4*C*rank;   // fcproj_A
        param_sizes[27] = L * rank*C;     // fcproj_B
    }
}

// 2. Optimized Forward Pass with Precomputed Weights
void matmul_forward_dora(float* out,
                        const float* inp, const float* weight, const float* bias,
                        DoRALayer* layer,
                        int B, int T, int C, int OC) {
    if (!layer) {
        matmul_forward(out, inp, weight, bias, B, T, C, OC);
        return;
    }

    // Precompute modified weights once
    float* modified_weights = (float*)malloc(OC * C * sizeof(float));
    const float eps = 1e-6f;

    #pragma omp parallel for collapse(2)
    for (int o = 0; o < OC; o++) {
        for (int i = 0; i < C; i++) {
            float mag = fmaxf(layer->magnitude[o], eps);
            float dir = weight[o*C + i] / mag;
            float lora = 0.0f;

            // Low-rank decomposition
            for (int r = 0; r < layer->rank; r++) {
                lora += layer->A[i*layer->rank + r] * layer->B[r*OC + o];
            }
            modified_weights[o*C + i] = mag * (dir + layer->alpha * lora);
        }
    }

    matmul_forward(out, inp, modified_weights, bias, B, T, C, OC);
    free(modified_weights);
}



// 3. Correct Gradient Accumulation in Backward Pass
void matmul_backward_dora(float* dinp, float* dweight, float* dbias,
                         float* dmagnitude, float* dA, float* dB,
                         const float* dout, const float* inp, const float* weight,
                         DoRALayer* layer,
                         int B, int T, int C, int OC) {
    if (!layer) {
        matmul_backward(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);
        return;
    }

    const int rank = layer->rank;
    const float alpha = layer->alpha;
    const float eps = 1e-6f;

    // Initialize gradients through OpenMP reduction
#pragma omp parallel for reduction(+:dmagnitude[:OC], dA[:C*rank], dB[:rank*OC])
    for (int bt = 0; bt < B*T; bt++) {
        const float* x = inp + bt * C;
        const float* dout_bt = dout + bt * OC;

        for (int o = 0; o < OC; o++) {
            float mag = fmaxf(layer->magnitude[o], eps);
            float dL_do = dout_bt[o];
            const float* weight_row = weight + o * C;

            // Magnitude gradient
            float sum_dir = 0.0f;
            for (int i = 0; i < C; i++) {
                sum_dir += x[i] * (weight_row[i] / mag);
            }
            dmagnitude[o] += dL_do * sum_dir;

            // Low-rank gradients
            for (int r = 0; r < rank; r++) {
                float a_grad = 0.0f, b_grad = 0.0f;
                for (int i = 0; i < C; i++) {
                    a_grad += x[i] * layer->B[r * OC + o];
                    b_grad += layer->A[i * rank + r] * x[i];
                }
                dA[o * rank + r] += alpha * dL_do * a_grad / mag;
                dB[r * OC + o] += alpha * dL_do * b_grad / mag;
            }

            // Input gradient
            for (int i = 0; i < C; i++) {
                dinp[bt * C + i] += dL_do * weight_row[i];
            }
        }
    }
}


// allocate memory for the parameters and point the individual tensors to the right places
#define NUM_PARAMETER_TENSORS 28

float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // Allocate all parameters at once.
    float* params_memory = (float*)mallocCheck(num_parameters * sizeof(float));
    // Create an array of pointers for all 28 tensors.
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b,
        &params->qkvw, &params->qkvb, &params->attprojw, &params->attprojb,
        &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb,
        &params->qkv_magnitude, &params->qkv_A, &params->qkv_B,
        &params->attproj_magnitude, &params->attproj_A, &params->attproj_B,
        &params->fc_magnitude, &params->fc_A, &params->fc_B,
        &params->fcproj_magnitude, &params->fcproj_A, &params->fcproj_B
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 23
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* qkv; // (L, B, T, 3*C)
    float* atty; // (L, B, T, C)
    float* preatt; // (L, B, NH, T, T)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* logits; // (B, T, V)
    float* probs; // (B, T, V)
    float* losses; // (B, T)
} ActivationTensors;

void fill_in_activation_sizes(size_t* act_sizes, GPT2Config config, int B, int T) {
    size_t C = config.channels;
    size_t NH = config.num_heads;
    size_t L = config.num_layers;
    size_t Vp = config.padded_vocab_size;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * 3 * C; // qkv
    act_sizes[5] = L * B * T * C; // atty
    act_sizes[6] = L * B * NH * T * T; // preatt
    act_sizes[7] = L * B * NH * T * T; // att
    act_sizes[8] = L * B * T * C; // attproj
    act_sizes[9] = L * B * T * C; // residual2
    act_sizes[10] = L * B * T * C; // ln2
    act_sizes[11] = L * B * T; // ln2_mean
    act_sizes[12] = L * B * T; // ln2_rstd
    act_sizes[13] = L * B * T * 4 * C; // fch
    act_sizes[14] = L * B * T * 4 * C; // fch_gelu
    act_sizes[15] = L * B * T * C; // fcproj
    act_sizes[16] = L * B * T * C; // residual3
    act_sizes[17] = B * T * C; // lnf
    act_sizes[18] = B * T; // lnf_mean
    act_sizes[19] = B * T; // lnf_rstd
    act_sizes[20] = B * T * Vp; // logits
    act_sizes[21] = B * T * Vp; // probs
    act_sizes[22] = B * T; // losses
}

float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory = (float*)mallocCheck(num_activations * sizeof(float));
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

typedef struct {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {
    // Open the checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);

    // Validate header
    if (model_header[0] != 20240326) {
        printf("Bad magic model file\n");
        exit(1);
    }
    if (model_header[1] != 3) {
        printf("Bad version in model file\n");
        exit(1);
    }

    // Read hyperparameters
    model->config.max_seq_len = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];
    model->config.use_dora = 1;
    model->config.dora_rank = 8;
    model->config.dora_alpha = 1.0f;

    // Print config
    printf("[GPT-2]\n");
    printf("max_seq_len: %zu\n", model->config.max_seq_len);
    printf("vocab_size: %zu\n", model->config.vocab_size);
    printf("padded_vocab_size: %zu\n", model->config.padded_vocab_size);
    printf("num_layers: %zu\n", model->config.num_layers);
    printf("num_heads: %zu\n", model->config.num_heads);
    printf("channels: %zu\n", model->config.channels);

    // Compute parameter sizes including DoRA parameters
    fill_in_parameter_sizes(model->param_sizes, model->config);

    // Allocate memory for ALL parameters (base + DoRA)
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);

    // Read base parameters from checkpoint
    size_t base_num_parameters = 0;
    for (int i = 0; i < 16; i++) {
        base_num_parameters += model->param_sizes[i];
    }
    freadCheck(model->params_memory, sizeof(float), base_num_parameters, model_file);
    fcloseCheck(model_file);

    // Initialize DoRA parameters if enabled
    if (model->config.use_dora) {
        const size_t L = model->config.num_layers;
        const size_t C = model->config.channels;
        const size_t rank = model->config.dora_rank;

        // Initialize each layer's DoRA components
        for (size_t l = 0; l < L; l++) {
            // Query-key-value projection
            dora_init(&(DoRALayer){
                .magnitude = &model->params.qkv_magnitude[l * 3*C],
                .A = &model->params.qkv_A[l * C*rank],
                .B = &model->params.qkv_B[l * rank*3*C],
                .rank = rank,
                .in_dim = C,
                .out_dim = 3*C
            }, C, 3*C, rank, model->config.dora_alpha,
            &model->params.qkvw[l * 3*C*C]);

            // Attention projection
            dora_init(&(DoRALayer){
                .magnitude = &model->params.attproj_magnitude[l*C],
                .A = &model->params.attproj_A[l*C*rank],
                .B = &model->params.attproj_B[l*rank*C],
                .rank = rank,
                .in_dim = C,
                .out_dim = C
            }, C, C, rank, model->config.dora_alpha,
            &model->params.attprojw[l*C*C]);

            // Feed-forward expansion
            dora_init(&(DoRALayer){
                .magnitude = &model->params.fc_magnitude[l*4*C],
                .A = &model->params.fc_A[l*C*rank],
                .B = &model->params.fc_B[l*rank*4*C],
                .rank = rank,
                .in_dim = C,
                .out_dim = 4*C
            }, C, 4*C, rank, model->config.dora_alpha,
            &model->params.fcw[l*4*C*C]);

            // Feed-forward projection
            dora_init(&(DoRALayer){
                .magnitude = &model->params.fcproj_magnitude[l*C],
                .A = &model->params.fcproj_A[l*4*C*rank],
                .B = &model->params.fcproj_B[l*rank*C],
                .rank = rank,
                .in_dim = 4*C,
                .out_dim = C
            }, 4*C, C, rank, model->config.dora_alpha,
            &model->params.fcprojw[l*C*4*C]);
        }
    }

    // Initialize remaining pointers
    model->num_parameters = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        model->num_parameters += model->param_sizes[i];
    }

    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f;
}

// --- Forward Pass ---
// --- Corrected Forward Pass Function ---
void gpt2_forward(GPT2 *model, int* inputs, int* targets, size_t B, size_t T) {
    if (model->params_memory == NULL) {
        printf("Error: model not initialized properly.\n");
        exit(1);
    }

    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;
    int use_dora = model->config.use_dora;

    // Validate inputs
    for (int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    // Lazy allocation of activation memory
    if (model->acts_memory == NULL) {
        model->batch_size = B;
        model->seq_len = T;
        fill_in_activation_sizes(model->act_sizes, model->config, B, T);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        model->inputs = (int*)mallocCheck(B * T * sizeof(int));
        model->targets = (int*)mallocCheck(B * T * sizeof(int));
    } else {
        if (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n",
                   model->batch_size, model->seq_len, (int)B, (int)T);
            exit(EXIT_FAILURE);
        }
    }

    // Cache inputs/targets
    memcpy(model->inputs, inputs, B * T * sizeof(int));
    if (targets != NULL) {
        memcpy(model->targets, targets, B * T * sizeof(int));
    }

    // Forward pass
    ParameterTensors params = model->params;
    ActivationTensors acts  = model->acts;
    float* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C);

    for (int l = 0; l < L; l++) {
        residual = (l == 0) ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;

        // Layer parameters
        float* l_ln1w     = params.ln1w     + l * C;
        float* l_ln1b     = params.ln1b     + l * C;
        float* l_qkvw     = params.qkvw     + l * 3 * C * C;
        float* l_qkvb     = params.qkvb     + l * 3 * C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w     = params.ln2w     + l * C;
        float* l_ln2b     = params.ln2b     + l * C;
        float* l_fcw      = params.fcw      + l * 4 * C * C;
        float* l_fcb      = params.fcb      + l * 4 * C;
        float* l_fcprojw  = params.fcprojw  + l * C * 4 * C;
        float* l_fcprojb  = params.fcprojb  + l * C;

        // DoRA parameters
        float* l_qkv_mag     = use_dora ? params.qkv_magnitude     + l * 3 * C : NULL;
        float* l_qkv_A       = use_dora ? params.qkv_A             + l * C * model->config.dora_rank : NULL;
        float* l_qkv_B       = use_dora ? params.qkv_B             + l * model->config.dora_rank * 3 * C : NULL;
        float* l_attproj_mag = use_dora ? params.attproj_magnitude + l * C : NULL;
        float* l_attproj_A   = use_dora ? params.attproj_A         + l * C * model->config.dora_rank : NULL;
        float* l_attproj_B   = use_dora ? params.attproj_B         + l * model->config.dora_rank * C : NULL;
        float* l_fc_mag      = use_dora ? params.fc_magnitude      + l * 4 * C : NULL;
        float* l_fc_A        = use_dora ? params.fc_A              + l * C * model->config.dora_rank : NULL;
        float* l_fc_B        = use_dora ? params.fc_B              + l * model->config.dora_rank * 4 * C : NULL;
        float* l_fcproj_mag  = use_dora ? params.fcproj_magnitude  + l * C : NULL;
        float* l_fcproj_A    = use_dora ? params.fcproj_A          + l * 4 * C * model->config.dora_rank : NULL;
        float* l_fcproj_B    = use_dora ? params.fcproj_B          + l * model->config.dora_rank * C : NULL;

        // Layer activations
        float* l_ln1       = acts.ln1       + l * B * T * C;
        float* l_ln1_mean  = acts.ln1_mean  + l * B * T;
        float* l_ln1_rstd  = acts.ln1_rstd  + l * B * T;
        float* l_qkv       = acts.qkv       + l * B * T * 3 * C;
        float* l_atty      = acts.atty      + l * B * T * C;
        float* l_preatt    = acts.preatt    + l * B * NH * T * T;
        float* l_att       = acts.att       + l * B * NH * T * T;
        float* l_attproj   = acts.attproj   + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2       = acts.ln2       + l * B * T * C;
        float* l_ln2_mean  = acts.ln2_mean  + l * B * T;
        float* l_ln2_rstd  = acts.ln2_rstd  + l * B * T;
        float* l_fch       = acts.fch       + l * B * T * 4 * C;
        float* l_fch_gelu  = acts.fch_gelu  + l * B * T * 4 * C;
        float* l_fcproj    = acts.fcproj    + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;

        // Forward operations with proper DoRA initialization
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);

        if (use_dora) {
            DoRALayer qkv_layer = construct_dora_layer(l_qkv_mag, l_qkv_A, l_qkv_B,
                                               C, 3*C, model->config.dora_rank, model->config.dora_alpha);
            matmul_forward_dora(l_qkv, l_ln1, l_qkvw, l_qkvb,
                                &qkv_layer,
                                B, T, C, 3*C);
        } else {
            matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        }

        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);

        if (use_dora) {
            DoRALayer attproj_layer = construct_dora_layer(l_attproj_mag, l_attproj_A, l_attproj_B,
                                                  C, C, model->config.dora_rank, model->config.dora_alpha);
            matmul_forward_dora(l_attproj, l_atty, l_attprojw, l_attprojb,
                                &attproj_layer,
                                B, T, C, C);
        } else {
            matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        }

        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);

        if (use_dora) {
            DoRALayer fc_layer = construct_dora_layer(l_fc_mag, l_fc_A, l_fc_B,
                                             C, 4*C, model->config.dora_rank, model->config.dora_alpha);
            matmul_forward_dora(l_fch, l_ln2, l_fcw, l_fcb,
                                &fc_layer,
                                B, T, C, 4*C);
        } else {
            matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        }

        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);

        if (use_dora) {
            DoRALayer fcproj_layer = construct_dora_layer(l_fcproj_mag, l_fcproj_A, l_fcproj_B,
                                                4*C, C, model->config.dora_rank, model->config.dora_alpha);
            matmul_forward_dora(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb,
                                &fcproj_layer,
                                B, T, 4*C, C);
        } else {
            matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        }

        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }

    residual = acts.residual3 + (L-1)*B*T*C;
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp);
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    if (targets != NULL) {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, Vp);
        float mean_loss = 0.0f;
        for (int i = 0; i < B*T; i++) {
            mean_loss += model->acts.losses[i];
        }
        model->mean_loss = mean_loss / (B*T);
    } else {
        model->mean_loss = -1.0f;
    }
}

void gpt2_zero_grad(GPT2 *model) {
    if(model->grads_memory != NULL) { memset(model->grads_memory, 0, model->num_parameters * sizeof(float)); }
    if(model->grads_acts_memory != NULL) { memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float)); }
}

// --- Backward Pass ---
void gpt2_backward(GPT2 *model) {
    if (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(1);
    }

    if (model->grads_memory == NULL) {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    size_t B = model->batch_size;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;
    int use_dora = model->config.use_dora;

    ParameterTensors params = model->params;
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // Initialize the loss gradients (each position gets 1/(B*T))
    float dloss_mean = 1.0f / (B * T);
    for (int i = 0; i < B * T; i++) {
        grads_acts.losses[i] = dloss_mean;
    }

    // Backprop through softmax + cross-entropy
    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses,
                                  acts.probs, model->targets, B, T, V, Vp);
    // Backprop through final linear: logits = lnf * wte
    matmul_backward(grads_acts.lnf, grads.wte, NULL,
                    grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    // Backprop through final layernorm (lnf)
    layernorm_backward(grads_acts.encoded, grads.lnfw, grads.lnfb,
                       grads_acts.lnf, acts.lnf, params.lnfw,
                       acts.lnf_mean, acts.lnf_rstd, B, T, C);

    // Now work backwards through the transformer layers
    for (int l = L - 1; l >= 0; l--) {
        // The gradient "dresidual" coming from the block above:
        float* dresidual = (l == 0) ? grads_acts.encoded :
                           grads_acts.residual3 + (l - 1) * B * T * C;
        // The forward-pass input to layer l (either encoder output or previous block)
        float* residual = (l == 0) ? acts.encoded :
                          acts.residual3 + (l - 1) * B * T * C;

        // ------------------------------
        // Get pointers to the activations from the forward pass (for layer l)
        float* l_ln1       = acts.ln1       + l * B * T * C;
        float* l_qkv       = acts.qkv       + l * B * T * 3 * C;
        float* l_atty      = acts.atty      + l * B * T * C;
        float* l_attproj   = acts.attproj   + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2       = acts.ln2       + l * B * T * C;
        float* l_fch       = acts.fch       + l * B * T * 4 * C;
        float* l_fch_gelu  = acts.fch_gelu  + l * B * T * 4 * C;
        float* l_fcproj    = acts.fcproj    + l * B * T * C;

        // Get pointers to layer parameters (forward)
        float* l_ln1w      = params.ln1w      + l * C;
        float* l_qkvw      = params.qkvw      + l * 3 * C * C;
        float* l_qkvb      = params.qkvb      + l * 3 * C;
        float* l_attprojw  = params.attprojw  + l * C * C;
        float* l_attprojb  = params.attprojb  + l * C;
        float* l_ln2w      = params.ln2w      + l * C;
        float* l_ln2b      = params.ln2b      + l * C;
        float* l_fcw       = params.fcw       + l * 4 * C * C;
        float* l_fcb       = params.fcb       + l * 4 * C;
        float* l_fcprojw   = params.fcprojw   + l * C * 4 * C;
        float* l_fcprojb   = params.fcprojb   + l * C;

        // DoRA parameters (if enabled)
        float* l_qkv_mag     = use_dora ? params.qkv_magnitude     + l * 3 * C : NULL;
        float* l_qkv_A       = use_dora ? params.qkv_A             + l * C * model->config.dora_rank : NULL;
        float* l_qkv_B       = use_dora ? params.qkv_B             + l * model->config.dora_rank * 3 * C : NULL;
        float* l_attproj_mag = use_dora ? params.attproj_magnitude + l * C : NULL;
        float* l_attproj_A   = use_dora ? params.attproj_A         + l * C * model->config.dora_rank : NULL;
        float* l_attproj_B   = use_dora ? params.attproj_B         + l * model->config.dora_rank * C : NULL;
        float* l_fc_mag      = use_dora ? params.fc_magnitude      + l * 4 * C : NULL;
        float* l_fc_A        = use_dora ? params.fc_A              + l * C * model->config.dora_rank : NULL;
        float* l_fc_B        = use_dora ? params.fc_B              + l * model->config.dora_rank * 4 * C : NULL;
        float* l_fcproj_mag  = use_dora ? params.fcproj_magnitude  + l * C : NULL;
        float* l_fcproj_A    = use_dora ? params.fcproj_A          + l * 4 * C * model->config.dora_rank : NULL;
        float* l_fcproj_B    = use_dora ? params.fcproj_B          + l * model->config.dora_rank * C : NULL;

        // ------------------------------
        // Get pointers to activation gradients for layer l (from grads_acts)
        float* dl_residual2 = grads_acts.residual2 + l * B * T * C;
        float* dl_fcproj    = grads_acts.fcproj    + l * B * T * C;
        float* dl_fch_gelu  = grads_acts.fch_gelu  + l * B * T * 4 * C;
        float* dl_fch       = grads_acts.fch       + l * B * T * 4 * C;
        float* dl_ln2       = grads_acts.ln2       + l * B * T * C;
        float* dl_attproj   = grads_acts.attproj   + l * B * T * C;
        float* dl_atty      = grads_acts.atty      + l * B * T * C;
        float* dl_qkv       = grads_acts.qkv       + l * B * T * 3 * C;
        float* dl_ln1       = grads_acts.ln1       + l * B * T * C;
        // For attention backward we also need the pre-attention and att gradients:
        float* dl_preatt    = grads_acts.preatt    + l * B * NH * T * T;
        float* dl_att       = grads_acts.att       + l * B * NH * T * T;

        // Get pointers to parameter gradients for layer l (from grads)
        float* dl_ln1w     = grads.ln1w     + l * C;
        float* dl_ln1b     = grads.ln1b     + l * C;
        float* dl_qkvw     = grads.qkvw     + l * 3 * C * C;
        float* dl_qkvb     = grads.qkvb     + l * 3 * C;
        float* dl_attprojw = grads.attprojw + l * C * C;
        float* dl_attprojb = grads.attprojb + l * C;
        float* dl_ln2w     = grads.ln2w     + l * C;
        float* dl_ln2b     = grads.ln2b     + l * C;
        float* dl_fcw      = grads.fcw      + l * 4 * C * C;
        float* dl_fcb      = grads.fcb      + l * 4 * C;
        float* dl_fcprojw  = grads.fcprojw  + l * C * 4 * C;
        float* dl_fcprojb  = grads.fcprojb  + l * C;

        // DoRA parameter gradients (if enabled)
        float* dl_qkv_mag     = use_dora ? grads.qkv_magnitude     + l * 3 * C : NULL;
        float* dl_qkv_A       = use_dora ? grads.qkv_A               + l * C * model->config.dora_rank : NULL;
        float* dl_qkv_B       = use_dora ? grads.qkv_B               + l * model->config.dora_rank * 3 * C : NULL;
        float* dl_attproj_mag = use_dora ? grads.attproj_magnitude   + l * C : NULL;
        float* dl_attproj_A   = use_dora ? grads.attproj_A           + l * C * model->config.dora_rank : NULL;
        float* dl_attproj_B   = use_dora ? grads.attproj_B           + l * model->config.dora_rank * C : NULL;
        float* dl_fc_mag      = use_dora ? grads.fc_magnitude        + l * 4 * C : NULL;
        float* dl_fc_A        = use_dora ? grads.fc_A                + l * C * model->config.dora_rank : NULL;
        float* dl_fc_B        = use_dora ? grads.fc_B                + l * model->config.dora_rank * 4 * C : NULL;
        float* dl_fcproj_mag  = use_dora ? grads.fcproj_magnitude    + l * C : NULL;
        float* dl_fcproj_A    = use_dora ? grads.fcproj_A            + l * 4 * C * model->config.dora_rank : NULL;
        float* dl_fcproj_B    = use_dora ? grads.fcproj_B            + l * model->config.dora_rank * C : NULL;

        // ------------------------------
        // Backprop through the residual addition that produced residual3:
        // forward: residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
        residual_backward(dl_residual2, dl_fcproj, dresidual, B * T * C);

        // Backprop through the fcproj linear layer (with DoRA if enabled)
        if (use_dora) {
            DoRALayer fcproj_layer = construct_dora_layer(l_fcproj_mag, l_fcproj_A, l_fcproj_B,
                                                          4 * C, C, model->config.dora_rank, model->config.dora_alpha);
            matmul_backward_dora(dl_fch_gelu, dl_fcprojw, dl_fcprojb,
                                 dl_fcproj_mag, dl_fcproj_A, dl_fcproj_B,
                                 dl_fcproj, l_fch_gelu, l_fcprojw,
                                 &fcproj_layer, B, T, 4 * C, C);
        } else {
            matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb,
                            dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4 * C, C);
        }

        // Backprop through GeLU activation
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * 4 * C);

        // Backprop through the first FC layer (fcw) with DoRA if enabled:
        if (use_dora) {
            DoRALayer fc_layer = construct_dora_layer(l_fc_mag, l_fc_A, l_fc_B,
                                                      C, 4 * C, model->config.dora_rank, model->config.dora_alpha);
            matmul_backward_dora(dl_ln2, dl_fcw, dl_fcb,
                                 dl_fc_mag, dl_fc_A, dl_fc_B,
                                 dl_fch, l_ln2, l_fcw,
                                 &fc_layer, B, T, C, 4 * C);
        } else {
            matmul_backward(dl_ln2, dl_fcw, dl_fcb,
                            dl_fch, l_ln2, l_fcw, B, T, C, 4 * C);
        }

        // Backprop through the layernorm (ln2) applied to residual2:
        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2,
                           l_residual2, l_ln2w,
                           acts.ln2_mean + l * B * T, acts.ln2_rstd + l * B * T,
                           B, T, C);

        // Backprop through the residual addition before the attention branch:
        // forward: residual_forward(l_residual2, residual, l_attproj, B*T*C);
        residual_backward(dresidual, dl_attproj, dl_residual2, B * T * C);

        // Backprop through the attention projection linear layer with DoRA if enabled:
        if (use_dora) {
            DoRALayer attproj_layer = construct_dora_layer(l_attproj_mag, l_attproj_A, l_attproj_B,
                                                           C, C, model->config.dora_rank, model->config.dora_alpha);
            matmul_backward_dora(dl_atty, dl_attprojw, dl_attprojb,
                                 dl_attproj_mag, dl_attproj_A, dl_attproj_B,
                                 dl_attproj, l_atty, l_attprojw,
                                 &attproj_layer, B, T, C, C);
        } else {
            matmul_backward(dl_atty, dl_attprojw, dl_attprojb,
                            dl_attproj, l_atty, l_attprojw, B, T, C, C);
        }

        // Backprop through the attention mechanism.
        // forward: attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty,
                           l_qkv, acts.att + l * B * NH * T * T, B, T, C, NH);

        // Backprop through the initial linear (qkv) with DoRA if enabled:
        if (use_dora) {
            DoRALayer qkv_layer = construct_dora_layer(l_qkv_mag, l_qkv_A, l_qkv_B,
                                                       C, 3 * C, model->config.dora_rank, model->config.dora_alpha);
            matmul_backward_dora(dl_ln1, dl_qkvw, dl_qkvb,
                                  dl_qkv_mag, dl_qkv_A, dl_qkv_B,
                                  dl_qkv, l_ln1, l_qkvw,
                                  &qkv_layer, B, T, C, 3 * C);
        } else {
            matmul_backward(dl_ln1, dl_qkvw, dl_qkvb,
                            dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C);
        }

        // Finally, backprop through the first layernorm (ln1)
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1,
                           residual, l_ln1w,
                           acts.ln1_mean + l * B * T, acts.ln1_rstd + l * B * T,
                           B, T, C);
    }

    // Backprop through the initial encoder embedding layer
    encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);
}


void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps,
                float weight_decay, int t) {
    // Calculate total parameter count including DoRA parameters
    size_t total_params = model->num_parameters;
    size_t dora_params_count = 0;

    if (model->config.use_dora) {
        // Count all DoRA parameters
        for (int i = 16; i < 28; i++) {
            dora_params_count += model->param_sizes[i];
        }
    }

    // Initialize Adam optimizer state if not already done
    if (model->m_memory == NULL) {
        // Allocate memory for base parameters
        model->m_memory = (float*)calloc(model->num_parameters, sizeof(float));
        model->v_memory = (float*)calloc(model->num_parameters, sizeof(float));

        // Allocate additional memory for DoRA parameters if needed
        if (model->config.use_dora) {
            // Extend memory for DoRA parameters
            model->m_memory = (float*)realloc(model->m_memory, (total_params + dora_params_count) * sizeof(float));
            model->v_memory = (float*)realloc(model->v_memory, (total_params + dora_params_count) * sizeof(float));

            // Initialize the new memory to zero
            memset(model->m_memory + total_params, 0, dora_params_count * sizeof(float));
            memset(model->v_memory + total_params, 0, dora_params_count * sizeof(float));
        }
    }

    // Update base parameters
    #pragma omp parallel for
    for (size_t i = 0; i < model->num_parameters; i++) {
        float param = model->params_memory[i];
        float grad = model->grads_memory[i];

        // AdamW update
        float m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
        float v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        model->m_memory[i] = m;
        model->v_memory[i] = v;
        model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }

    // Update DoRA parameters if enabled
    if (model->config.use_dora) {
        // Define arrays of DoRA parameter pointers and their gradient pointers
        float* dora_params[] = {
            model->params.qkv_magnitude, model->params.qkv_A, model->params.qkv_B,
            model->params.attproj_magnitude, model->params.attproj_A, model->params.attproj_B,
            model->params.fc_magnitude, model->params.fc_A, model->params.fc_B,
            model->params.fcproj_magnitude, model->params.fcproj_A, model->params.fcproj_B
        };

        float* dora_grads[] = {
            model->grads.qkv_magnitude, model->grads.qkv_A, model->grads.qkv_B,
            model->grads.attproj_magnitude, model->grads.attproj_A, model->grads.attproj_B,
            model->grads.fc_magnitude, model->grads.fc_A, model->grads.fc_B,
            model->grads.fcproj_magnitude, model->grads.fcproj_A, model->grads.fcproj_B
        };

        // Iterate through each DoRA parameter type
        size_t m_offset = model->num_parameters; // Start after base parameters

        for (int i = 0; i < 12; i++) {
            size_t param_size = model->param_sizes[i + 16]; // DoRA params start at index 16

            if (param_size == 0) continue; // Skip if this DoRA parameter isn't used

            // Apply AdamW updates to each parameter in this DoRA component
            for (size_t j = 0; j < param_size; j++) {
                float param = dora_params[i][j];
                float grad = dora_grads[i][j];

                // AdamW update
                float m = beta1 * model->m_memory[m_offset] + (1.0f - beta1) * grad;
                float v = beta2 * model->v_memory[m_offset] + (1.0f - beta2) * grad * grad;
                float m_hat = m / (1.0f - powf(beta1, t));
                float v_hat = v / (1.0f - powf(beta2, t));

                model->m_memory[m_offset] = m;
                model->v_memory[m_offset] = v;

                // Apply lower weight decay to DoRA parameters (optional)
                float dora_weight_decay = weight_decay * 0.1f; // Reduced weight decay for DoRA

                // Update the parameter
                dora_params[i][j] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + dora_weight_decay * param);

                m_offset++;
            }
        }
    }
}

void gpt2_free(GPT2 *model) {
    free(model->params_memory);
    free(model->grads_memory);
    free(model->m_memory);
    free(model->v_memory);
    free(model->acts_memory);
    free(model->grads_acts_memory);
    free(model->inputs);
    free(model->targets);
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

unsigned int random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(uint64_t *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
int main() {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    const char* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    const char* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    const char* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    const char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    int B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
    printf("train dataset num_batches: %zu\n", train_loader.num_tokens / (B*T));
    printf("val dataset num_batches: %zu\n", val_loader.num_tokens / (B*T));
    int val_num_batches = 5;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    uint64_t rng_state = 1337;
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    const int genT = 64; // number of steps of inference we will do

    // train
    struct timespec start, end;
    for (int step = 0; step <= 40; step++) {

        // once in a while estimate the validation loss
        if (step % 10 == 0) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        if (step > 0 && step % 20 == 0) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            for(int i = 0; i < B * T; ++i) {
                gen_tokens[i] = tokenizer.eot_token;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            for (int t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_tokens, NULL, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                float* probs = model.acts.probs + (t-1) * model.config.padded_vocab_size;
                float coin = random_f32(&rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                int next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
    }

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    free(gen_tokens);
    return 0;
}
#endif

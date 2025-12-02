# !pip install BPEmb
import math
import numpy as np
import tensorflow as tf

# from bpemb import BPEmb

# Transformers From Scratch
# We'll build a transformer from scratch, layer-by-layer. We'll start with the Multi-Head Self-Attention layer since that's the most involved bit. Once we have that working, the rest of the model will look familiar if you've been following the course so far.

# Multi-Head Self-Attention

# Scaled Dot Product Self-Attention

# Inside each attention head is a Scaled Dot Product Self-Attention operation as we covered in the slides. Given queries, keys, and values, the operation returns a new "mix" of the values.

# Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

# The following function implements this and also takes a mask to account for padding and for masking future tokens for decoding (i.e. look-ahead mask). We'll cover masking later in the notebook.
def scaled_dot_product_attention(query, key, value, mask=None):
    key_dim = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_scores = tf.matmul(query, key, transpose_b=True) / np.sqrt(key_dim)

    if mask is not None:
        scaled_scores = tf.where(mask == 0, -np.inf, scaled_scores)

    softmax = tf.keras.layers.Softmax()
    weights = softmax(scaled_scores)
    return tf.matmul(weights, value), weights

# Suppose our queries, keys, and values are each a length of 3 with a dimension of 4.
seq_len = 3
embed_dim = 4

queries = np.random.rand(seq_len, embed_dim)
keys = np.random.rand(seq_len, embed_dim)
values = np.random.rand(seq_len, embed_dim)

print("Queries:\n", queries)

# This would be the self-attention output and weights.
output, attn_weights = scaled_dot_product_attention(queries, keys, values)

print("Output\n", output, "\n")
print("Weights\n", attn_weights)

# Generating queries, keys, and values for multiple heads.

# Now that we have a way to calculate self-attention, let's actually generate the input queries, keys, and values for multiple heads.

# In the slides (and in most references), each attention head had its own separate set of query, key, and value weights. Each weight matrix was of dimension d x d/h where h was the number of heads.

# It's easier to understand things this way and we can certainly code it this way as well. But we can also "simulate" different heads with a single query matrix, single key matrix, and single value matrix.
# We'll do both. First we'll create query, key, and value vectors using separate weights per head.

batch_size = 1
seq_len = 3
embed_dim = 12
num_heads = 3
head_dim = embed_dim // num_heads

print(f"Dimension of each head: {head_dim}")

# Using separate weight matrices per head
# Suppose these are our input embeddings. Here we have a batch of 1 containing a sequence of length 3, with each element being a 12-dimensional embedding.
x = np.random.rand(batch_size, seq_len, embed_dim).round(1)
print("Input shape ", x.shape)
print("Input:\n", x)

# We'll declare three sets of query weights one for each head, three sets of key weights, and three sets of value weights. Remember each weight matrix should have a dimension of d x dh.

# The query weights for each head.
wq0 = np.random.rand(embed_dim, head_dim).round(1)
wq1 = np.random.rand(embed_dim, head_dim).round(1)
wq2 = np.random.rand(embed_dim, head_dim).round(1)

# The key weights for each head.
wk0 = np.random.rand(embed_dim, head_dim).round(1)
wk1 = np.random.rand(embed_dim, head_dim).round(1)
wk2 = np.random.rand(embed_dim, head_dim).round(1)

# The value weights for each head.
wv0 = np.random.rand(embed_dim, head_dim).round(1)
wv1 = np.random.rand(embed_dim, head_dim).round(1)
wv2 = np.random.rand(embed_dim, head_dim).round(1)

print("The three sets of query weights one for each head...\n")
print("wq0:\n", wq0)
print("wq1:\n", wq1)
print("wq2:\n", wq1)

# We'll generate our queries, keys, and values for each head by multiplying our input by the weights.
# Generated queries, keys, and values for the first head.
q0 = np.dot(x, wq0)
k0 = np.dot(x, wk0)
v0 = np.dot(x, wv0)

# Generated queries, keys, and values for the second head.
q1 = np.dot(x, wq1)
k1 = np.dot(x, wk1)
v1 = np.dot(x, wv1)

# Generated queries, keys, and values for the third head.
q2 = np.dot(x, wq2)
k2 = np.dot(x, wk2)
v2 = np.dot(x, wv2)

print("Q, K, and V for first head:\n")
print(f"q0 {q0.shape}:\n", q0)
print(f"k0 {k0.shape}:\n", k0)
print(f"v0 {v0.shape}:\n", v0)

# Now that we have our Q, K, V vectors, we can just pass them to our self-attention operation. Here we're calculating the output and attention weights for the first head.
out0, attn_weights0 = scaled_dot_product_attention(q0, k0, v0)

print("Output from first attention head:\n", out0)
print("Attention weights from first head:\n", attn_weights0)

# Here are the other two attention weights ignored.
out1, _ = scaled_dot_product_attention(q1, k1, v1)
out2, _ = scaled_dot_product_attention(q2, k2, v2)

print("Output from second attention head:\n", out1)
print("Output from third attention head:\n", out2)

# As we covered in the slides, once we have each head's output, we concatenate them and then put them through a linear layer for further processing.
combined_out_a = np.concatenate((out0, out1, out2), axis=-1)
print(f"Combined output from all heads: {combined_out_a.shape}\n", combined_out_a)

# The final step would be to run combined_out_a through a linear/dense layer, for further processing.

# So that's a complete run of multi-head self-attention using separate sets of weights per head.

# Let's now get the same thing done using a single query weight matrix, single key weight matrix, and single value weight matrix.

# These were our separate per-head query weights
print("Query weights for first head:\n", wq0)
print("Query weights for second head:\n", wq1)
print("Query weights for third head:\n", wq2)

# Suppose instead of declaring three separate query weight matrices, we had declared one. i.e. a single d x d matrix. We're concatenating our per-head query weights here instead of declaring a new set of weights so that we get the same results.
wq = np.concatenate((wq0, wq1, wq2), axis=1)
print(f"Single query weight matrix: {wq.shape}\n", wq)

# In the same vein, pretend we declared a single key weight matrix, and single value weight matrix.
wk = np.concatenate((wk0, wk1, wk2), axis=1)
wv = np.concatenate((wv0, wv1, wv2), axis=1)

print(f"Single key weight matrix: {wk.shape}\n", wk)
print(f"Single value weight matrix: {wv.shape}\n", wv)

# Now we can calculate all our queries, keys, and values with three dot products.
qs = np.dot(x, wq)
ks = np.dot(x, wk)
vs = np.dot(x, wv)

# These are our resulting query vectors we'll call them combined queries. How do we simulate different heads with this?
print(f"Query vectors using a single weight matrix: {qs.shape}\n", qs)

# Somehow, we need to separate these vectors such they're treated like three separate sets by the self-attention operation.
print("q0\n", q0)
print("q1\n", q1)
print("q2\n", q2)

# Notice how each set of per-head queries looks like we took the combined queries, and chopped them vertically every four dimensions.

# We can split our combined queries into d x dh heads using reshape and transpose.

# The first step is to reshape our combined queries from a shape of
# batch_size, seq_len, embed_dim
# into a shape of
# batch_size, seq_len, num_heads, head_dim.

# Note we can achieve the same thing by passing -1 instead of seq_len.
qs_reshaped = tf.reshape(qs, (batch_size, seq_len, num_heads, head_dim))
print(f"Combined queries: {qs.shape}\n", qs)

print(f"Reshaped into separate heads: {qs_reshaped.shape}\n", qs_reshaped)

# At this point, we have our desired shape. The next step is to transpose it such that simulates vertically chopping our combined queries. By transposing, our matrix dimensions become
# batch_size, num_heads, seq_len, head_dim
qs_transposed = tf.transpose(qs_reshaped, perm=[0, 2, 1, 3]).numpy()
print(f"Queries transposed into separate heads: {qs_transposed.shape},\n", qs_transposed)

# If we compare this against the separate per-head queries we calculated previously, we see the same result except we now have all our queries in a single matrix.
print("The separate per-head query matrices from before:\n")
print("q0\n", q0)
print("q1\n", q1)
print("q2\n", q2)

# Let's do the exact same thing with our combined keys and values.
ks_transposed = tf.transpose(tf.reshape(ks, (batch_size, -1, num_heads, head_dim)), perm=[0, 2, 1, 3]).numpy()
vs_transposed = tf.transpose(tf.reshape(vs, (batch_size, -1, num_heads, head_dim)), perm=[0, 2, 1, 3]).numpy()

print(f"Keys for all heads in a single matrix: {ks.shape}\n", ks_transposed)
print(f"Values for all heads in a single matrix: {vs.shape}\n", vs_transposed)

# Set up this way, we can now calculate the outputs from all attention heads with a single call to our self-attention operation.
all_heads_output, all_attn_weights = scaled_dot_product_attention(qs_transposed,
                                                                  ks_transposed,
                                                                  vs_transposed)
print("Self attention output:\n", all_heads_output)

# As a sanity check, we can compare this against the outputs from individual heads we calculated earlier
print("Per head outputs from using separate sets of weights per head\n")
print("out0\n", out0)
print("out1\n", out1)
print("out2\n", out2)

# To get the final concatenated result, we need to reverse our reshape and transpose operation, starting with the transpose this time.
combined_out_b = tf.reshape(tf.transpose(all_heads_output, perm=[0, 2, 1, 3]),
                            shape=(batch_size, seq_len, embed_dim))
print("Final output from using single query, key, value matrices,\n", combined_out_b)

print("Final output from using separate query, key, value matrices per head,\n", combined_out_a)

# We can encapsulate everything we just covered in a class.
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_head = self.d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(self.d_model)
        self.wk = tf.keras.layers.Dense(self.d_model)
        self.wv = tf.keras.layers.Dense(self.d_model)

        # Linear layer to generate the final output.
        self.dense = tf.keras.layers.Dense(self.d_model)

    def split_heads(self, x):
        batch_size = x.shape[0]

        split_inputs = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_head))
        return tf.transpose(split_inputs, perm=[0, 2, 1, 3])

    def merge_heads(self, x):
        batch_size = x.shape[0]

        merged_inputs = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(merged_inputs, (batch_size, -1, self.d_model))

    def call(self, q, k, v, mask):
        qs = self.wq(q)
        ks = self.wk(k)
        vs = self.wv(v)

        qs = self.split_heads(qs)
        ks = self.split_heads(ks)
        vs = self.split_heads(vs)

        output, attn_weights = scaled_dot_product_attention(qs, ks, vs, mask)
        output = self.merge_heads(output)

        return self.dense(output), attn_weights

mhsa = MultiHeadSelfAttention(12, 3)

output, attn_weights = mhsa(x, x, x, None)
print(f"MHSA output: {output.shape}\n", output)

# Encoder Block

# We can now build our Encoder Block. In addition to the Multi-Head Self Attention layer, the Encoder Block also has skip connections, layer normalization steps, and a two-layer feed-forward neural network.

# The original Attention Is All You Need paper also included some dropout applied to the self-attention output which isn't shown in the illustration below see references for a link to the paper.

# Since a two-layer feed forward neural network is used in multiple places in the transformer, here's a function which creates and returns one.
def feed_forward_network(d_model, hidden_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_dim, activation='relu'),
        tf.keras.layers.Dense(d_model),
    ])

# This is our encoder block containing all the layers and steps from the preceding illustration plus dropout.
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, hidden_dim, dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.mhsa = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, hidden_dim)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()

    def call(self, x, training, mask):
        mhsa_output, attn_weights = self.mhsa(x, x, x, mask)
        mhsa_output = self.dropout1(mhsa_output, training=training)
        mhsa_output = self.layer_norm1(x + mhsa_output)

        ffn_output = self.ffn(mhsa_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        output = self.layer_norm2(mhsa_output + ffn_output)

        return output, attn_weights

# Suppose we have an embedding dimension of 12, and we want 3 attention heads and a feed forward network with a hidden dimension of 48 (4x the embedding dimension). We would declare and use a single encoder block like so
encoder_block = EncoderBlock(12, 3, 48)

block_output, _ = encoder_block(x, True, None)
print(f"Output from single encoder block: {block_output.shape}\n", block_output)

# Word and Positional Embeddings
# Let's now deal with the actual input to the initial encoder block. The inputs are going to be positional word embeddings. That is, word embeddings with some positional information added to them.

# Let's start with subword tokenization. For demonstration, we'll use a subword tokenizer called BPEmb. It uses Byte-Pair Encoding and supports over two hundred languages.

# Load the English tokenizer.
# bpemb_en = BPEmb(lang="en")

# The library comes with embeddings for a number of words.
bpemb_vocab_size, bpemb_embed_size = bpemb_en.vectors.shape
print("Vocabulary size:", bpemb_vocab_size)
print("Embedding size:", bpemb_embed_size)

# We don't need the embeddings since we're going to use our own embedding layer. What we're interested in are the subword tokens and their respective ids. The ids will be used as indexes into our embedding layer.

# These are the subword tokens for our example sentence from the slides. BPEmb places underscores in front of any tokens which are whole words or intended to begin words.

# Remember that subword tokenizers are trained using count frequencies over a corpus. So these subword tokens are specific to BPEmb. Another subword tokenizer may output something different. This is why it's important that when we use a pretrained model, we make sure to use the pretrained model's tokenizer. We'll see this when we use pretrained transformers later in this module.

sample_sentence = "Where can I find a pizzeria?"
tokens = bpemb_en.encode(sample_sentence)
print(tokens)

# We can retrieve each subword token's respective id using the encode_ids method.
token_seq = np.array(bpemb_en.encode_ids(sample_sentence))
print(token_seq)

# Now that we have a way to tokenize and vectorize sentences, we can declare and use an embedding layer with the same vocabulary size as BPEmb and a desired embedding size.

embed_dim = 12
token_embed = tf.keras.layers.Embedding(bpemb_vocab_size, embed_dim)
token_embeddings = token_embed(token_seq)

# The untrained embeddings for our sample sentence.
print("Embeddings for ", sample_sentence)
print(token_embeddings)

# Next, we need to add positional information to each token embedding. As we covered in the slides, the original paper used sinusoidals but it's more common these days to just use another set of embeddings. We'll do the latter here.

# Here, we're declaring an embedding layer with rows equalling a maximum sequence length and columns equalling our token embedding size. We then generate a vector of position ids.

max_seq_len = 256
pos_embed = tf.keras.layers.Embedding(max_seq_len, embed_dim)

# Generate ids for each position of the token sequence.
pos_idx = tf.range(len(token_seq))
print(pos_idx)

# We'll use these position ids to index into the positional embedding layer.

# These are our position embeddings.
position_embeddings = pos_embed(pos_idx)
print("Position embeddings for the input sequence:\n", position_embeddings)

# The final step is to add our token and position embeddings. The result will be the input to the first encoder block.

input_embed = token_embeddings + position_embeddings
print("Input to the initial encoder block:\n", input_embed)

# Encoder

# Now that we have an encoder block and a way to embed our tokens with position information, we can create the encoder itself.

# Given a batch of vectorized sequences, the encoder creates positional embeddings, runs them through its encoder blocks, and returns contextualized tokens.

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_blocks, d_model, num_heads, hidden_dim, src_vocab_size,
                 max_seq_len, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = tf.keras.layers.Embedding(src_vocab_size, self.d_model)
        self.pos_embed = tf.keras.layers.Embedding(max_seq_len, self.d_model)

        # The original Attention Is All You Need paper applied dropout to the
        # input before feeding it to the first encoder block.
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # Create encoder blocks.
        self.blocks = [EncoderBlock(self.d_model, num_heads, hidden_dim, dropout_rate)
                       for _ in range(num_blocks)]

    def call(self, input, training, mask):
        token_embeds = self.token_embed(input)

        # Generate position indices for a batch of input sequences.
        num_pos = input.shape[0] * self.max_seq_len
        pos_idx = np.resize(np.arange(self.max_seq_len), num_pos)
        pos_idx = np.reshape(pos_idx, input.shape)
        pos_embeds = self.pos_embed(pos_idx)

        x = self.dropout(token_embeds + pos_embeds, training=training)

        # Run input through successive encoder blocks.
        for block in self.blocks:
            x, weights = block(x, training, mask)

        return x, weights

# If you're wondering about this code block here,
# ```
# num_pos = input.shape * self.max_seq_len
# pos_idx = np.resize(np.arange(self.max_seq_len), num_pos)
# pos_idx = np.reshape(pos_idx, input.shape)
# pos_embeds = self.pos_embed(pos_idx)
# ```
# This generates positional embeddings for a batch of input sequences. Suppose this was our batch of input sequences to the encoder.

# Batch of 3 sequences, each of length 10 (10 is also the maximum sequence length in this case).
seqs = np.random.randint(0, 10000, size=(3, 10))
print("seqs.shape:", seqs.shape)
print(seqs)

# We need to retrieve a positional embedding for every element in this batch. The first step is to create the respective positional ids...
pos_ids = np.resize(np.arange(seqs.shape[1]), seqs.shape[0] * seqs.shape[1])
print(pos_ids)

# ...and then reshape them to match the input batch dimensions.
pos_ids = np.reshape(pos_ids, (3, 10))
print("pos_ids.shape:", pos_ids.shape)
print(pos_ids)

# We can now retrieve position embeddings for every token embedding.
pos_embed(pos_ids)

# Let's try our encoder on a batch of sentences.
input_batch = [
    "Where can I find a pizzeria?",
    "Mass hysteria over listeria.",
    "I ain't no circle back girl.",
]

input_batch_encoded = bpemb_en.encode(input_batch)

input_seqs = bpemb_en.encode_ids(input_batch)
print("Vectorized inputs:\n", input_seqs)

# Note how the input sequences aren't the same length in this batch. In this case, we need to pad them out so that they are. If you're unfamiliar with why, refer to the notebook on Recurrent Neural Networks
# https://colab.research.google.com/github/futuremojo/nlp-demystified/blob/main/notebooks/nlp-demystified-recurrent-neural-networks.ipynb

# We'll do this using pad_sequences.
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences

padded_input_seqs = tf.keras.preprocessing.sequence.pad_sequences(input_seqs, padding='post')
print("Input to the encoder:")
print("padded_input_seqs.shape:", padded_input_seqs.shape)
print(padded_input_seqs)

# Since our input now has padding, now's a good time to cover masking.

# So given a mask, wherever there's a mask position set to 0, the corresponding position in the attention scores will be set to -inf. The resulting attention weight for the position will then be zero and no attending will occur for that position.

# In the slides, we covered look-ahead masks for the decoder to prevent it from attending to future tokens, but we also need masks for padding.

# In total, there are three masks involved:
# 1. The encoder mask to mask out any padding in the encoder sequences.
# 2. The decoder mask which is used in the decoder's first multi-head self-attention layer. It's a combination of two masks one to account for the padding in target sequences, and the look-ahead mask.
# 3. The memory mask which is used in the decoder's second multi-head self-attention layer. The keys and values for this layer are going to be the encoder's output, and this mask will ensure the decoder doesn't attend to any encoder output which corresponds to padding. In practice, 1 and 3 are often the same.

# The scaled_dot_product_attention function has this line,
# if mask is not None:
#     scaled_scores = tf.where(mask==0, -np.inf, scaled_scores)

# Let's create an encoder mask for our batch of input sequences.
# Wherever there's padding, we want the mask position set to zero.
enc_mask = tf.cast(tf.math.not_equal(padded_input_seqs, 0), tf.float32)
print("Input:\n", padded_input_seqs)
print("Encoder mask:\n", enc_mask)

# Keep in mind that the dimension of the attention matrix for this example is going to be
# batch size, number of heads, query size, key size
# (3, 3, 10, 10)

# So we need to expand the mask dimensions like so
enc_mask = enc_mask[:, tf.newaxis, tf.newaxis, :]
print(enc_mask)

# This way, the encoder mask will now be broadcasted.
# https://www.tensorflow.org/xla/broadcasting

# Now we can declare an encoder and pass it batches of vectorized sequences.
num_encoder_blocks = 6

# d_model is the embedding dimension used throughout.
d_model = 12

num_heads = 3

# Feed-forward network hidden dimension width.
ffn_hidden_dim = 48

src_vocab_size = bpemb_vocab_size
max_input_seq_len = padded_input_seqs.shape[1]

encoder = Encoder(
    num_encoder_blocks,
    d_model,
    num_heads,
    ffn_hidden_dim,
    src_vocab_size,
    max_input_seq_len
)

# We can now pass our input sequences and mask to the encoder.
encoder_output, attn_weights = encoder(padded_input_seqs, training=True,
                                       mask=enc_mask)
print(f"Encoder output: {encoder_output.shape}\n", encoder_output)

# Decoder Block

# Let's build the Decoder Block. Everything we did to create the encoder block applies here. The major differences are that the Decoder Block has...
# 1. a Multi-Head Cross-Attention layer which uses the encoder's outputs as the keys and values.
# 2. an extra skip/residual connection along with an extra layer normalization step.

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, hidden_dim, dropout_rate=0.1):
        super(DecoderBlock, self).__init__()

        self.mhsa1 = MultiHeadSelfAttention(d_model, num_heads)
        self.mhsa2 = MultiHeadSelfAttention(d_model, num_heads)

        self.ffn = feed_forward_network(d_model, hidden_dim)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()
        self.layer_norm3 = tf.keras.layers.LayerNormalization()

        # Note the decoder block takes two masks. One for the first MHSA, another
        # for the second MHSA.

    def call(self, encoder_output, target, training, decoder_mask, memory_mask):
        mhsa_output1, attn_weights = self.mhsa1(target, target, target, decoder_mask)
        mhsa_output1 = self.dropout1(mhsa_output1, training=training)
        mhsa_output1 = self.layer_norm1(mhsa_output1 + target)

        mhsa_output2, attn_weights = self.mhsa2(mhsa_output1, encoder_output,
                                                 encoder_output,
                                                 memory_mask)
        mhsa_output2 = self.dropout2(mhsa_output2, training=training)
        mhsa_output2 = self.layer_norm2(mhsa_output2 + mhsa_output1)

        ffn_output = self.ffn(mhsa_output2)
        ffn_output = self.dropout3(ffn_output, training=training)
        output = self.layer_norm3(ffn_output + mhsa_output2)

        return output, attn_weights

# Decoder

# The decoder is almost the same as the encoder except it takes the encoder's output as part of its input, and it takes two masks the decoder mask and memory mask.
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_blocks, d_model, num_heads, hidden_dim, target_vocab_size,
                 max_seq_len, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = tf.keras.layers.Embedding(target_vocab_size, self.d_model)
        self.pos_embed = tf.keras.layers.Embedding(max_seq_len, self.d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.blocks = [DecoderBlock(self.d_model, num_heads, hidden_dim, dropout_rate)
                       for _ in range(num_blocks)]

    def call(self, encoder_output, target, training, decoder_mask, memory_mask):
        token_embeds = self.token_embed(target)

        # Generate position indices.
        num_pos = target.shape[0] * self.max_seq_len
        pos_idx = np.resize(np.arange(self.max_seq_len), num_pos)
        pos_idx = np.reshape(pos_idx, target.shape)

        pos_embeds = self.pos_embed(pos_idx)

        x = self.dropout(token_embeds + pos_embeds, training=training)

        for block in self.blocks:
            x, weights = block(encoder_output, x, training, decoder_mask, memory_mask)

        return x, weights

# Before we try the decoder, let's cover the masks involved. The decoder takes two masks:
# The decoder mask which is a combination of two masks one to account for the padding in target sequences, and the look-ahead mask. This mask is used in the decoder's first multi-head self-attention layer.
# The memory mask which is used in the decoder's second multi-head self-attention. The keys and values for this layer are going to be the encoder's output, and this mask will ensure the decoder doesn't attend to any encoder output which corresponds to padding.

# Suppose this is our batch of vectorized target input sequences for the decoder. These values are just made up.
# Note If you need a refresher on how to prepare target input and output sequences for the decoder, refer to the seq2seq notebook.

# Made up values.
target_input_seqs = [
    [1, 652, 723, 123, 62],
    [1, 25, 98, 129, 248, 215, 359, 249],
    [1, 2369, 1259, 125, 486],
]

# As we did with the encoder input sequences, we need to pad out this batch so that all sequences within it are the same length.
padded_target_input_seqs = tf.keras.preprocessing.sequence.pad_sequences(target_input_seqs, padding='post')
print("Padded target inputs to the decoder:")
print("padded_target_input_seqs.shape:", padded_target_input_seqs.shape)
print(padded_target_input_seqs)

# We can create the padding mask the same way we did for the encoder.
dec_padding_mask = tf.cast(tf.math.not_equal(padded_target_input_seqs, 0), tf.float32)
dec_padding_mask = dec_padding_mask[:, tf.newaxis, tf.newaxis, :]

print(dec_padding_mask)

# As we covered in the slides, the look-ahead mask is a diagonal where the lower half are 1s and the upper half are zeros. This is easy to create using the bandpart method
# https://www.tensorflow.org/api_docs/python/tf/linalg/band_part

target_input_seq_len = padded_target_input_seqs.shape[1]
lookahead_mask = tf.linalg.band_part(tf.ones((target_input_seq_len, target_input_seq_len)), -1, 0)
print(lookahead_mask)

# To create the decoder mask, we just need to combine the padding and look-ahead masks. Note how the columns of the resulting decoder mask are all zero for padding positions.
dec_mask = tf.minimum(dec_padding_mask, lookahead_mask)
print("The decoder mask:\n", dec_mask)

# We can now declare a decoder and pass it everything it needs. In our case, the memory mask is the same as the encoder mask.

decoder = Decoder(6, 12, 3, 48, 10000, 8)
decoder_output, _ = decoder(encoder_output, padded_target_input_seqs,
                            True, dec_mask, enc_mask)
print(f"Decoder output: {decoder_output.shape}\n", decoder_output)

# Transformer

# We now have all the pieces to build the Transformer itself, and it's pretty simple.
class Transformer(tf.keras.Model):
    def __init__(self, num_blocks, d_model, num_heads, hidden_dim, source_vocab_size,
                 target_vocab_size, max_input_len, max_target_len, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_blocks, d_model, num_heads, hidden_dim, source_vocab_size,
                              max_input_len, dropout_rate)

        self.decoder = Decoder(num_blocks, d_model, num_heads, hidden_dim, target_vocab_size,
                              max_target_len, dropout_rate)

        # The final dense layer to generate logits from the decoder output.
        self.output_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, input_seqs, target_input_seqs, training, encoder_mask,
             decoder_mask, memory_mask):
        encoder_output, encoder_attn_weights = self.encoder(input_seqs,
                                                            training, encoder_mask)

        decoder_output, decoder_attn_weights = self.decoder(encoder_output,
                                                            target_input_seqs, training,
                                                            decoder_mask, memory_mask)

        return self.output_layer(decoder_output), encoder_attn_weights, decoder_attn_weights

transformer = Transformer(
    num_blocks=6,
    d_model=12,
    num_heads=3,
    hidden_dim=48,
    source_vocab_size=bpemb_vocab_size,
    target_vocab_size=7000,  # made-up target vocab size.
    max_input_len=padded_input_seqs.shape[1],
    max_target_len=padded_target_input_seqs.shape[1],
)

transformer_output, _, _ = transformer(padded_input_seqs,
                                      padded_target_input_seqs, True,
                                      enc_mask, dec_mask, memory_mask=enc_mask)
print(f"Transformer output: {transformer_output.shape}\n", transformer_output)
print("If training, we would use this output to calculate losses.")

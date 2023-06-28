from module import swin_1d_block
from examples import (
    random_text_generator,
    generate_random_dna,
    onehot_encoder,
)


def test_genomic_model(seq_length=512):
    input = generate_random_dna(seq_length)
    encode_input = onehot_encoder(input)
    model = swin1d_block(4)
    output = model(encode_input)
    print(output.shape)
    return output


def test_language_model(seq_length=512):
    input = random_text_generator(2, seq_length, tokenized=True)
    model = swin1d_block(1)
    output = model(input)
    print(output.shape)
    return output


def swin1d_block(dim):
    # stage = (number layers in each swin,
    #          whether to merge the ouput of each swin,
    #          window size)

    window_size = 32
    stages = [
        (
            4,
            True,
            window_size,
        ),
        (
            2,
            False,
            window_size,
        ),
        (
            2,
            False,
            window_size,
        ),
        (
            2,
            False,
            window_size,
        ),
    ]
    model = swin_1d_block(stages, dim)
    return model


if __name__ == "__main__":
    test_language_model()

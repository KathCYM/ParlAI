import os
import torch
import numpy as np
from parlai.chat_service.services.browser_chat.paraphraser.para import init_gpt2_model
# from para import init_gpt2_model
import time
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def preprocess_input_sent(in_sent, tokenizer, max_prefix_length):
    # 1. Convert sentence to tokens
    in_tokens = np.array(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(in_sent)), dtype=np.int32)
    # 2. Truncate sentence if too long
    if len(in_tokens) > max_prefix_length:
        print("Input too long, truncated")
        in_tokens = in_tokens[:max_prefix_length]
    # 3. Pad sentence if sentence is shorter than max_prefix_length
    num_pad = max_prefix_length - len(in_tokens)
    in_tokens = np.pad(in_tokens, (num_pad, 0), constant_values=tokenizer.pad_token_id)
    sentence = np.concatenate([in_tokens, [tokenizer.bos_token_id]]) # Add bos to trigger output generation
    # 4. Create segment (token_type_ids)
    in_segment = [tokenizer.additional_special_tokens_ids[1] for _ in in_tokens]
    segment = np.concatenate([in_segment, [tokenizer.additional_special_tokens_ids[2]]]).astype(np.int64)
    # 5. Transfer everything to tensor, unsqueeze to resember a batch of size 1
    sentence = torch.tensor(sentence).unsqueeze(0)
    segment = torch.tensor(segment).unsqueeze(0)
    # 6. Done, return
    return sentence, segment

def output_to_text(output, init_context_size, tokenizer):
    # cut the output sentence out of the entire token sequence
    output_seq = output[init_context_size:].tolist()
    if tokenizer.eos_token_id in output_seq:
        output_seq = output_seq[:output_seq.index(tokenizer.eos_token_id)]
    # return decoded sentence
    return tokenizer.decode(output_seq, clean_up_tokenization_spaces=True, skip_special_tokens=True)

def make_inference(in_sent, tokenizer, gpt2_model):
    max_prefix_length = 50 # Max length for input sentence so far
    beam_size = 3 # Beam size used in beam search

    # preprocess
    sentence, segment = preprocess_input_sent(in_sent, tokenizer, max_prefix_length)
    sentence = sentence.to(device)
    segment = segment.to(device)
    init_context_size = max_prefix_length + 1 # +1 for bos token

    # make inference
    out, dense_length, scores = gpt2_model.generate(gpt2_sentences=sentence,
                                                    segments=segment,
                                                    eos_token_id=tokenizer.eos_token_id,
                                                    init_context_size=init_context_size,
                                                    beam_size=beam_size)
    
    generated_text = output_to_text(out[0], init_context_size, tokenizer)
    # print("Generated paraphrase: {}\n".format(generated_text))
    return generated_text

def get_one_sentence(input_phrase):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    # Load model
    load_chkpt_step = 21918
    load_path = os.path.join(os.getcwd(), 'paraphraser', 'pretrain', 'chkpt-{}'.format(load_chkpt_step))
    # load_path = os.path.join('pretrain', 'chkpt-{}'.format(load_chkpt_step))
    print("````````````````````````````````````````loading from " + load_path)
    # print(load_path)
    gpt2_model, tokenizer = init_gpt2_model(load_path, device, True)
    config = gpt2_model.gpt2.config
    t1 = time.time()
    print("t1-t0: " + str(t1-t0))
    # Make inference
    split_msgs = re.split('[;.?]', input_phrase)
    print("split_msgs " +  str(split_msgs))
    # split_msgs = input_phrase.split('.')
    ret = ""
    for msg in split_msgs:
        print( msg)
        if msg == "":
            continue
        ret += make_inference(msg, tokenizer, gpt2_model) + " " 
    return ret

if __name__ == '__main__':
    input_phrase = "hi , how are you today ? i just got back from a long day of work , how about you ?"
    print("you called me?")
    print(get_one_sentence(input_phrase))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # t0 = time.time()
    # # Load model
    # load_chkpt_step = 21918
    # load_path = os.path.join('pretrain', 'chkpt-{}'.format(load_chkpt_step))
    # # print(load_path)
    # gpt2_model, tokenizer = init_gpt2_model(load_path, device, True)
    # config = gpt2_model.gpt2.config
    # t1 = time.time()
    # print("t1-t0: " + str(t1-t0))
    # # Make inference
    # while True:
    #     sentence = input('Enter a sentence: ')
    #     if sentence == 'END':
    #         print("Goodbye!")
    #         break
    #     make_inference(sentence, tokenizer)
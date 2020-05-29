#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2020/5/22 12:54
# @Author  : Cathy 
# @FileName: translate.py

import torch
# import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from torchvision import transforms
from src.model import make_model, beam_search_decode, EncoderDecoder, greedy_decode
from PIL import Image
import json
# import jieba

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(src_ori):
    # Load vocabulary JSON
    maxlen = 10
    nbest = 1
    pad = 0

    try:
        voc = json.load(open('src/vocab/tran_vocab.json', 'rb'))
    except:
        raise IOError("Please generate tran_vocab.json")

    # Build the model
    model = make_model(len(voc['src']), len(voc['trg'])).to(device)
    # Load pretrained model
    model.load_state_dict(torch.load('src/model/translateModel.pth.tar', map_location='cpu'))
    # model = torch.load('src/model/translateModel.pth.tar', map_location='cpu')
    # they behave differently in evaluation mode

    # Transfer model to gpu or stay in cpu
    model.to(device)

    model.eval()

    with torch.no_grad():

        # prepare src src_mask
        src_list = src_ori[:-1].split()
        src = list()
        for s in src_list:
            try:
                src.append(voc['src'][s])
            except:
                src.append(voc['src']['<unk>'])
        src = torch.IntTensor(src).to(torch.int64).unsqueeze(-2)
        # print(src != 0)
        src_mask = (src != pad).unsqueeze(-3).int()

        vocablist = sorted(voc['trg'].keys(), key=lambda s: voc['trg'][s])

        # Generate an caption from the image
        if True:
            pred_out, _ = beam_search_decode(model, src, src_mask, maxlen, start_symbol=voc['trg']['<sos>'],
                                             unk_symbol=voc['trg']['<unk>'], end_symbol=voc['trg']['<eos>'],
                                             pad_symbol=voc['trg']['<blank>'])

            for n in range(min(nbest, len(pred_out))):
                pred = pred_out[n]
                hypstr = []
                for w in pred[0]:
                    if w == voc['trg']['<eos>']:
                        break
                    hypstr.append(vocablist[w])
                hypstr = "".join(hypstr)
                print('HYP[%d]: %s  ( %f )' % (n + 1, hypstr, pred[1]))
        else:
            output = greedy_decode(model, src, src_mask, maxlen, start_symbol=voc['trg']['<sos>'])
            output = [i for i in output[0].cpu().numpy()]
            hypstr = []
            for i in output[1:]:
                if i == voc['trg']['<eos>']:
                    break
                hypstr.append(vocablist[i])
            hypstr = ''.join(hypstr)
            print('HYP: {}'.format(hypstr))

        return hypstr + '。'


if __name__ == '__main__':
    main('a person in the park.')


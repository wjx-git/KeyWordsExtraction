import argparse
import os
import json

import torch
from pytorch_pretrained_bert import BertAdam
from NegSamplingNER.utils import UnitAlphabet, LabelAlphabet, LexicalAlphabet
from NegSamplingNER.model import PhraseClassifier
from NegSamplingNER.misc import fix_random_seed
from NegSamplingNER.utils import corpus_to_iterator, Procedure


def inference(args):
    test_loader = corpus_to_iterator(os.path.join(args.data_dir, "test_roberta_top1.json"), args.batch_size, False)
    model = torch.load(os.path.join(args.check_dir, 'model.pt'))
    Procedure.test(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", "-dd", type=str, default='NegSamplingNER/data')
    parser.add_argument("--check_dir", "-cd", type=str, default='NegSamplingNER/save')
    parser.add_argument("--resource_dir", "-rd", type=str, default='NegSamplingNER/resource')
    parser.add_argument("--random_state", "-rs", type=int, default=0)
    parser.add_argument("--epoch_num", "-en", type=int, default=40)
    parser.add_argument("--batch_size", "-bs", type=int, default=8)

    parser.add_argument("--negative_rate", "-nr", type=int, default=0.7, help="positive sample : negative sample = 1:nr")
    parser.add_argument("--warmup_proportion", "-wp", type=float, default=0.1)
    parser.add_argument("--hidden_dim", "-hd", type=int, default=512)
    parser.add_argument("--dropout_rate", "-dr", type=float, default=0.1)

    parser.add_argument("--model", type=str, default='bert')

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=True), end="\n\n")

    fix_random_seed(args.random_state)

    label_vocab = LabelAlphabet()
    train_loader = corpus_to_iterator(os.path.join(args.data_dir, "train_roberta_top1.json"), args.batch_size, True, label_vocab)
    dev_loader = corpus_to_iterator(os.path.join(args.data_dir, "dev_roberta_top1.json"), args.batch_size, False)
    # test_loader = corpus_to_iterator(os.path.join(args.data_dir, "test.json"), args.batch_size, False)
    # test_un_loader = corpus_to_iterator(os.path.join(args.data_dir, "test_un_correct.json"), args.batch_size, False)

    total_steps = int(len(train_loader) * (args.epoch_num + 1))

    lexical_vocab = UnitAlphabet(os.path.join(args.resource_dir, "bert", "vocab.txt"))
    bert_path = os.path.join(args.resource_dir, "bert")
    model = PhraseClassifier(lexical_vocab, label_vocab, args.hidden_dim,
                             args.dropout_rate, args.negative_rate, bert_path)
    all_parameters = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_param = [
        {'params': [p for n, p in all_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in all_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(grouped_param, lr=1e-5, warmup=args.warmup_proportion, t_total=total_steps)

    model = model.cuda() if torch.cuda.is_available() else model.cpu()

    if not os.path.exists(args.check_dir):
        os.makedirs(args.check_dir)
    best_dev = 0.0
    script_path = os.path.join(args.resource_dir, "conlleval.pl")
    checkpoint_path = os.path.join(args.check_dir, "model.pt")

    for epoch_i in range(0, args.epoch_num + 1):
        loss, train_time = Procedure.train(model, train_loader, optimizer)
        # print("[Epoch {:3d}] loss on train set is {:.5f} using {:.3f} secs".format(epoch_i, loss, train_time))

        dev_f1, dev_time = Procedure.dev(model, dev_loader, script_path, epoch_i)
        print("(Epoch {:3d}) f1 score on dev set is {:.5f} using {:.3f} secs".format(epoch_i, dev_f1, dev_time))

        # dev_f1, dev_time = Procedure.test(model, test_un_loader, script_path, epoch_i)
        # print("(Epoch {:3d}) f1 score on dev set is {:.5f} using {:.3f} secs".format(epoch_i, dev_f1, dev_time))

        if dev_f1 > best_dev:
            best_dev = dev_f1

            # print("\n<Epoch {:3d}> save best model with score: {:.5f} in terms of test set".format(epoch_i, dev_f1))
            torch.save(model, checkpoint_path)

            # dev_f1, dev_time = Procedure.test(model, test_loader, script_path, epoch_i)
            # print("(Epoch {:3d}) f1 score on test set is {:.5f} using {:.3f} secs".format(epoch_i, dev_f1, dev_time))
            # dev_f1, dev_time = Procedure.test(model, test_un_loader, script_path, epoch_i)
            # print("(Epoch {:3d}) f1 score on test_un_correct is {:.5f} using {:.3f} secs".format(epoch_i, dev_f1, dev_time))

        # print(end="\n\n")
    # test_f1, test_time = Procedure.test(model, test_loader, script_path, epoch_i)
    # print("{{Epoch {:3d}}} f1 score on test set is {:.5f} using {:.3f} secs".format(epoch_i, test_f1, test_time))

    inference(args)

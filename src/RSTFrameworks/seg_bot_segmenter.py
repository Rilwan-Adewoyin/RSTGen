import sys
import re
from nltk.tokenize import word_tokenize
import pickle
import numpy as np

import torch
from models.SegBot.solver import TrainSolver
from models import SegBot

import torch.nn as nn
import torch.nn.utils.rnn as R
import torch.nn.functional as F
from torch.autograd import Variable

class PointerNetworks(nn.Module):
    def __init__(self, voca_size, voc_embeddings, word_dim, hidden_dim, is_bi_encoder_rnn, rnn_type, rnn_layers,
                 dropout_prob, use_cuda, finedtuning, isbanor):
        super(PointerNetworks, self).__init__()

        self.word_dim = word_dim
        self.voca_size = voca_size

        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.is_bi_encoder_rnn = is_bi_encoder_rnn
        self.num_rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.voc_embeddings = voc_embeddings
        self.finedtuning = finedtuning

        self.nnDropout = nn.Dropout(dropout_prob)

        self.isbanor = isbanor

        if rnn_type in ['LSTM', 'GRU']:

            self.decoder_rnn = getattr(nn, rnn_type)(input_size=word_dim,
                                                     hidden_size=2 * hidden_dim if is_bi_encoder_rnn else hidden_dim,
                                                     num_layers=rnn_layers,
                                                     dropout=dropout_prob,
                                                     batch_first=True)

            self.encoder_rnn = getattr(nn, rnn_type)(input_size=word_dim,
                                                     hidden_size=hidden_dim,
                                                     num_layers=rnn_layers,
                                                     bidirectional=is_bi_encoder_rnn,
                                                     dropout=dropout_prob,
                                                     batch_first=True)

        else:
            print('rnn_type should be LSTM,GRU')

        self.nnSELU = nn.SELU()

        self.nnEm = nn.Embedding(self.voca_size, self.word_dim)

        self.initEmbeddings(self.voc_embeddings)

        self.use_cuda = use_cuda
        self.use_cuda = False
        
        if self.is_bi_encoder_rnn:
            self.num_encoder_bi = 2
        else:
            self.num_encoder_bi = 1

        self.nnW1 = nn.Linear(self.num_encoder_bi * hidden_dim,
                              self.num_encoder_bi * hidden_dim, bias=False)
        self.nnW2 = nn.Linear(self.num_encoder_bi * hidden_dim,
                              self.num_encoder_bi * hidden_dim, bias=False)
        self.nnV = nn.Linear(self.num_encoder_bi * hidden_dim, 1, bias=False)

        if isbanor:
            self.bn_inputdata = nn.BatchNorm1d(
                self.word_dim, affine=False, track_running_stats=True)

    def initEmbeddings(self, weights):
        self.nnEm.weight.data.copy_(torch.from_numpy(weights))
        self.nnEm.weight.requires_grad = self.finedtuning

    def initHidden(self, hsize, batchsize, device='cpu'):

        if self.rnn_type == 'LSTM':
            h_0 = Variable(torch.zeros(self.num_encoder_bi *
                           self.num_rnn_layers, batchsize, hsize,device=device))
            c_0 = Variable(torch.zeros(self.num_encoder_bi *
                           self.num_rnn_layers, batchsize, hsize, device=device))

            return (h_0, c_0)
        else:

            h_0 = Variable(torch.zeros(self.num_encoder_bi *
                           self.num_rnn_layers, batchsize, hsize, device=device))

            return h_0

    def _run_rnn_packed(self, cell, x, x_lens, h=None):
        x_packed = R.pack_padded_sequence(x, x_lens.data.tolist(),
                                          batch_first=True)

        if h is not None:
            output, h = cell(x_packed, h)
        else:
            output, h = cell(x_packed)

        output, _ = R.pad_packed_sequence(output, batch_first=True)

        return output, h

    def pointerEncoder(self, Xin, lens):

        batch_size, maxL = Xin.size()

        X = self.nnEm(Xin)  # N L  C

        if self.isbanor and maxL > 1:
            X = X.permute(0, 2, 1)  # N C L
            X = self.bn_inputdata(X)
            X = X.permute(0, 2, 1)  # N L C

        X = self.nnDropout(X)

        encoder_lstm_co_h_o = self.initHidden(self.hidden_dim, batch_size, device = X.device)
        o, h = self._run_rnn_packed(
            self.encoder_rnn, X, lens, encoder_lstm_co_h_o)  # batch_first=True
        o = o.contiguous()

        o = self.nnDropout(o)

        return o, h

    def pointerLayer(self, en, di):
        """

        :param en:  [L,H]
        :param di:  [H,]
        :return:
        """

        WE = self.nnW1(en)

        exdi = di.expand_as(en)

        WD = self.nnW2(exdi)

        nnV = self.nnV(self.nnSELU(WE+WD))

        nnV = nnV.permute(1, 0)

        nnV = self.nnSELU(nnV)

        # TODO: for log loss
        att_weights = F.softmax(nnV, dim=-1)
        logits = F.log_softmax(nnV, dim=-1)

        return logits, att_weights

    def training_decoder(self, hn, hend, X, Xindex, Yindex, lens):
        """


        """

        loss_function = nn.NLLLoss()
        batch_loss = 0
        LoopN = 0
        batch_size = len(lens)
        for i in range(len(lens)):  # Loop batch size

            curX_index = Xindex[i]
            curY_index = Yindex[i]
            curL = lens[i]
            curX = X[i]

            x_index_var = Variable(
                torch.from_numpy(curX_index.astype(np.int64)))
            if self.use_cuda:
                x_index_var = x_index_var.cuda()

            cur_lookup = curX[x_index_var]

            curX_vectors = self.nnEm(cur_lookup)  # output: [seq,features]

            curX_vectors = curX_vectors.unsqueeze(0)  # [batch, seq, features]

            if self.rnn_type == 'LSTM':  # need h_end,c_end

                h_end = hend[0].permute(1, 0, 2).contiguous().view(
                    batch_size, self.num_rnn_layers, -1)
                c_end = hend[1].permute(1, 0, 2).contiguous().view(
                    batch_size, self.num_rnn_layers, -1)

                curh0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                curc0 = c_end[i].unsqueeze(0).permute(1, 0, 2)

                h_pass = (curh0, curc0)
            else:

                h_end = hend.permute(1, 0, 2).contiguous().view(
                    batch_size, self.num_rnn_layers, -1)
                curh0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                h_pass = curh0

            decoder_out, _ = self.decoder_rnn(curX_vectors, h_pass)
            decoder_out = decoder_out.squeeze(0)  # [seq,features]

            # hn[batch,seq,H] -->[seq,H] i is loop batch size
            curencoder_hn = hn[i, 0:curL, :]

            for j in range(len(decoder_out)):  # Loop di
                cur_dj = decoder_out[j]
                cur_groundy = curY_index[j]

                cur_start_index = curX_index[j]
                predict_range = list(range(cur_start_index, curL))

                # TODO: make it point backward, only consider predict_range in current time step
                # align groundtruth
                cur_groundy_var = Variable(torch.LongTensor(
                    [int(cur_groundy) - int(cur_start_index)]))
                if self.use_cuda:
                    cur_groundy_var = cur_groundy_var.cuda()

                curencoder_hn_back = curencoder_hn[predict_range, :]

                cur_logists, cur_weights = self.pointerLayer(
                    curencoder_hn_back, cur_dj)

                batch_loss = batch_loss + \
                    loss_function(cur_logists, cur_groundy_var)
                LoopN = LoopN + 1

        batch_loss = batch_loss/LoopN

        return batch_loss

    def neg_log_likelihood(self, Xin, index_decoder_x, index_decoder_y, lens):
        '''
        :param Xin:  stack_x, [allseq,wordDim]
        :param Yin:
        :param lens:
        :return:
        '''

        encoder_hn, encoder_h_end = self.pointerEncoder(Xin, lens)

        loss = self.training_decoder(
            encoder_hn, encoder_h_end, Xin, index_decoder_x, index_decoder_y, lens)

        return loss

    def test_decoder(self, hn, hend, X, Yindex, lens):

        loss_function = nn.NLLLoss()
        batch_loss = 0
        LoopN = 0

        batch_boundary = []
        batch_boundary_start = []
        batch_align_matrix = []

        batch_size = len(lens)

        for i in range(len(lens)):  # Loop batch size

            curL = lens[i]
            curY_index = Yindex[i]
            curX = X[i]
            cur_end_boundary = curY_index[-1]

            cur_boundary = []
            cur_b_start = []
            cur_align_matrix = []

            cur_sentence_vectors = self.nnEm(curX)  # output: [seq,features]

            if self.rnn_type == 'LSTM':  # need h_end,c_end

                h_end = hend[0].permute(1, 0, 2).contiguous().view(
                    batch_size, self.num_rnn_layers, -1)
                c_end = hend[1].permute(1, 0, 2).contiguous().view(
                    batch_size, self.num_rnn_layers, -1)

                curh0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                curc0 = c_end[i].unsqueeze(0).permute(1, 0, 2)

                h_pass = (curh0, curc0)
            else:  # only need h_end

                h_end = hend.permute(1, 0, 2).contiguous().view(
                    batch_size, self.num_rnn_layers, -1)
                curh0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                h_pass = curh0

            # hn[batch,seq,H] --> [seq,H]  i is loop batch size
            curencoder_hn = hn[i, 0:curL, :]

            Not_break = True

            loop_in = cur_sentence_vectors[0, :].unsqueeze(
                0).unsqueeze(0)  # [1,1,H]
            loop_hc = h_pass

            loopstart = torch.full( (1,) , 0,  dtype=torch.long, device=curencoder_hn.device )[0]

            loop_j = 0
            while (Not_break):  # if not end

                loop_o, loop_hc = self.decoder_rnn(loop_in, loop_hc)

                # TODO: make it point backward

                predict_range = list(range(loopstart, curL))
                curencoder_hn_back = curencoder_hn[predict_range, :]
                cur_logists, cur_weights = self.pointerLayer(
                    curencoder_hn_back, loop_o.squeeze(0).squeeze(0))

                cur_align_vector = np.zeros(curL)
                cur_align_vector[predict_range] = cur_weights.data.cpu().numpy()[
                    0]
                cur_align_matrix.append(cur_align_vector)

                # TODO:align groundtruth
                if loop_j > len(curY_index)-1:
                    cur_groundy = curY_index[-1]
                else:
                    cur_groundy = curY_index[loop_j]

                cur_groundy_var = Variable( torch.full( (1,),
                    max(0, int(cur_groundy) - loopstart), dtype=torch.long,device=cur_logists.device))
                
                batch_loss = batch_loss + \
                    loss_function(cur_logists, cur_groundy_var)

                # TODO: get predicted boundary
                topv, topi = cur_logists.data.topk(1)

                pred_index = topi[0][0]

                # TODO: align pred_index to original seq
                ori_pred_index = pred_index + loopstart

                if cur_end_boundary == ori_pred_index:
                    cur_boundary.append(ori_pred_index)
                    cur_b_start.append(loopstart)
                    Not_break = False
                    loop_j = loop_j + 1
                    LoopN = LoopN + 1
                    break
                else:
                    cur_boundary.append(ori_pred_index)

                    loop_in = cur_sentence_vectors[ori_pred_index +
                                                   1, :].unsqueeze(0).unsqueeze(0)
                    cur_b_start.append(loopstart)

                    loopstart = ori_pred_index+1  # start =  pred_end + 1

                    loop_j = loop_j + 1
                    LoopN = LoopN + 1

            # For each instance in batch
            batch_boundary.append(cur_boundary)
            batch_boundary_start.append(cur_b_start)
            batch_align_matrix.append(cur_align_matrix)

        batch_loss = batch_loss / LoopN

        if batch_boundary[0][0].device != 'cpu':
            batch_boundary =  torch.stack( [ torch.stack(bb) for bb in batch_boundary ], axis=0 ).cpu().numpy()
            batch_boundary_start =  torch.stack( [ torch.stack(bbs) for bbs in batch_boundary_start ], axis=0 ).cpu().numpy()

        else:
            batch_boundary = np.array(batch_boundary)
            batch_boundary_start = np.array(batch_boundary_start)
            batch_align_matrix = np.array(batch_align_matrix)

        return batch_loss, batch_boundary, batch_boundary_start, batch_align_matrix

    def predict(self, Xin, index_decoder_y, lens):

        batch_size = index_decoder_y.shape[0]

        encoder_hn, encoder_h_end = self.pointerEncoder(Xin, lens)

        batch_loss, batch_boundary, batch_boundary_start, batch_align_matrix = self.test_decoder(
            encoder_hn, encoder_h_end, Xin, index_decoder_y, lens)

        return batch_loss, batch_boundary, batch_boundary_start, batch_align_matrix

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"RE_DIGITS":1,"UNKNOWN":2,"PADDING":0}
        self.word2count = {"RE_DIGITS":1,"UNKNOWN":1,"PADDING":1}
        self.index2word = {0: "PADDING", 1: "RE_DIGITS", 2: "UNKNOWN"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.strip('\n').strip('\r').split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class Segmenter():
    
    def __init__(self, device='cpu'):

        all_voc = r'./models/SegBot/all_vocabulary.pickle'
        

        voca = pickle.load(open(all_voc, 'rb'))
        self.voca_dict = voca.word2index

        model = PointerNetworks(voca_size = len(self.voca_dict), voc_embeddings=np.ndarray(shape=(len(self.voca_dict),300), dtype=float),
                                     word_dim=300, hidden_dim=64, is_bi_encoder_rnn=True,rnn_type='GRU',
                                     rnn_layers=6,
                                     dropout_prob=0.5,use_cuda=False,finedtuning=True,isbanor=True)
        
        # loaded_model = torch.load(r'./models/SegBot/trained_model.torchsave', map_location=lambda storage, loc: storage)
        # torch.save(loaded_model, 'models/SegBot/segbot_model.pt') 
        # sys.modules['model'] = sys.modules["__main__"]
        loaded_model = torch.load('./models/SegBot/segbot_model.pt' ) 
        
        _ = model.load_state_dict( loaded_model.state_dict(), strict=False )
        model.eval()

        if next(model.parameters()).device.type != device:
            model.to(device) 
        
        self.solver = TrainSolver(model, '', '', '', '', '',
                           batch_size=1, eval_size=1, epoch=10, lr=1e-2, lr_decay_epoch=1, weight_decay=1e-4,
                           use_cuda=False)
              
            
    def tokenize(self, inS ):
        repDig = re.sub(r'\d*[\d,]*\d+', 'RE_DIGITS', inS)
        toked = word_tokenize(repDig)
        or_toked = word_tokenize(inS)
        re_unk_list = []
        ori_list = []

        for (i,t) in enumerate(toked):
            if t not in self.voca_dict and t not in ['RE_DIGITS']:
                re_unk_list.append('UNKNOWN')
                ori_list.append(or_toked[i])
            else:
                re_unk_list.append(t)
                ori_list.append(or_toked[i])


        labey_edus = [0]*len(re_unk_list)
        labey_edus[-1] = 1

        return ori_list,re_unk_list,labey_edus

    def get_mapping( self, X, Y):
        
        X_map = []
        for w in X:
            if w in self.voca_dict:
                X_map.append(self.voca_dict[w])
            else:
                X_map.append(self.voca_dict['UNKNOWN'])

        X_map = np.array([X_map])
        Y_map = np.array([Y])
        
        return X_map, Y_map

    def segment_li_utterances(self, li_utterance):
        
        tpl_utterance_seg = [None] * len(li_utterance)
        
        li_ori_X = [None]*len(li_utterance)
        
        li_X_in = [None]*len(li_utterance)
        li_Y_in = [None]*len(li_utterance)
        
        
        for idx, utt in enumerate(li_utterance):
            ori_X, X, Y = self.tokenize(utt)
            X_in, Y_in = self.get_mapping(X, Y)
            
            li_ori_X[idx] = ori_X
            li_X_in[idx] = np.squeeze(X_in, axis=0)
            li_Y_in[idx] = np.squeeze(Y_in,axis=0)
            
        # X_in_batched = np.concatenate(li_X_in, axis=0)
        # Y_in_batched = np.concatenate(li_Y_in, axis=0)
        
        test_batch_ave_loss, test_pre, test_rec, test_f1, visdata = self.solver.check_accuracy(li_X_in, li_Y_in)

        for idx in range(len(li_utterance)):
            starts_b = visdata[3][idx]
            ends_b = visdata[2][idx] + 1
            segments = []

            for START, END in zip(starts_b, ends_b):
                segments.append(' '.join(li_ori_X[idx][START:END]))

            tpl_utterance_seg[idx] = segments
            
        return tpl_utterance_seg

    def to(self, device):
        self.solver.model.to(device)

    
# pickler._Unpickler.find_class =
     
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):

        if name == "PointerNetworks":
            return PointerNetworks
            
        return super(RenameUnpickler, self).find_class(module, name)

if __name__ == '__main__':



    sent='Sheraton and Pan Am said they are assured under the Soviet joint-venture law that they can repatriate profits from their hotel venture.'

    li_edus_short = ["Does she happen to be Asian?", "I notice them doing this a lot,", "my wife included.", "I was thinking", "it may have something to do with the way", "they swarm vs. Queue in some countries.", "The ones", "I see blocking", "are of course not always Asian but they very frequently are."]
    sent_short = ' '.join(li_edus_short)

    li_edus_long = ["Talking to your girl would be the best way.", "But franklY.I suspect", "she's going through something a lot worse", "that you are,", "and it's all your fault", "she's feeling that way.", "Expatiate your guilt", "by being the best boyfriend", "you can be.", "If she leaves you,", "accept it was your fault,", "and try", "and make it up to the next girl", "by not doing whatever mysterious nonsense", "you got into.", "Related note :", "` Mistakes happen'?", "Dropping the milk is a mistake.", "Forgetting to pick up the dry cleaning is a mistake.", "Cheating, or doing some other preplanned act", "that you know", "is going to cause severe emotional distress to your significant other", "isn't a mistake.", "It's wrong.", "It was thought out,", "you knew", "it would hurt them,", "and you decided,", "you made a conscious choice", "to HURT them.", "No mere mistake thiS.A greater evil is afoot."]
    sent_long = ' '.join(li_edus_long)
    
    seg = Segmenter()
    
    seg.to('cuda:1')
    output = seg.segment_li_utterances([sent_short, sent_long])
    seg.to('cpu')
    
    
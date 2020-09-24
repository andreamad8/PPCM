import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import os
from transformers import BertTokenizer,BertModel,AutoConfig

EPSILON = 1e-10

class Discriminator(torch.nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(
            self,
            class_size,
            pretrained_model="medium",
            cached_mode=False,
            load_weight=None, 
            model_pretrained=None,
            entailment=False, 
            device='cuda'
    ):
        super(Discriminator, self).__init__()

        self.entailment = entailment
        model_path = f'models/dialoGPT/{pretrained_model}/'
        config = GPT2Config.from_json_file(os.path.join(model_path, 'config.json'))
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        if model_pretrained != None:
            self.encoder = model_pretrained
        else:
            self.encoder = load_model(GPT2LMHeadModel(config), model_path+f"{pretrained_model}_ft.pkl", None, verbose=True)
        self.embed_size = config.n_embd
        
        if(self.entailment):
            self.classifier_head = AttentionHead(class_size=class_size, embed_size=self.embed_size)
        else:
            self.classifier_head = ClassificationHead(
                class_size=class_size,
                embed_size=self.embed_size
            )
        self.cached_mode = cached_mode
        if(load_weight != None):
            self.classifier_head.load_state_dict(torch.load(load_weight))
        self.device = device
        self.class_size = class_size

    def get_classifier(self):
        return self.classifier_head

    def train_custom(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier_head.train()

    def avg_representation(self, x, entailment=False):
        mask = x.ne(0).unsqueeze(2).repeat(
            1, 1, self.embed_size
        ).float().to(self.device).detach()
        hidden, _ = self.encoder.transformer(x)
        masked_hidden = hidden * mask
        if(entailment):
            return masked_hidden
        else:
            avg_hidden = torch.sum(masked_hidden, dim=1) / (
                    torch.sum(mask, dim=1).detach() + EPSILON
            )
            return avg_hidden

    def forward(self, x):
        if(self.entailment):
            P = self.avg_representation(x[0].to(self.device),entailment=True)
            H = self.avg_representation(x[1].to(self.device),entailment=True)
            logits = self.classifier_head(P,H)
            return logits
        else:
            if self.cached_mode:
                avg_hidden = x.to(self.device)
            else:
                avg_hidden = self.avg_representation(x.to(self.device))

            logits = self.classifier_head(avg_hidden)
            return logits


class Scorer(torch.nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(
            self,
            hidden_dim,
            output_dim,
            n_layers,
            bidirectional,
            dropout
    ):
        super(Scorer, self).__init__()

        # self.entailment = entailment
        # self.class_size = class_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        embedding_dim = self.bert.config.to_dict()['hidden_size']
 
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded = self.bert(text)[0]
                
        #embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        
        #output = [batch size, out dim]
        
        return output


class AttentionHead(nn.Module):
    '''
        https://github.com/libowen2121/SNLI-decomposable-attention/blob/master/models/baseline_snli.py
        intra sentence attention
    '''

    def __init__(self, embed_size, class_size):
        super(AttentionHead, self).__init__()

        self.embed_size = embed_size
        self.class_size = class_size

        self.mlp_f = self._mlp_layers(self.embed_size, self.embed_size)
        self.mlp_g = self._mlp_layers(2 * self.embed_size, self.embed_size)
        self.mlp_h = self._mlp_layers(2 * self.embed_size, self.embed_size)

        self.final_linear = nn.Linear(self.embed_size, self.class_size)

    def _mlp_layers(self, input_dim, output_dim):
        mlp_layers = []
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(input_dim, output_dim))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(output_dim, output_dim))
        mlp_layers.append(nn.ReLU())        
        return nn.Sequential(*mlp_layers)   # * used to unpack list

    def forward(self, sent1_linear, sent2_linear):
        '''
            sent_linear: batch_size x length x hidden_size
        '''
        len1 = sent1_linear.size(1)
        len2 = sent2_linear.size(1)

        '''attend'''
        f1 = self.mlp_f(sent1_linear.view(-1, self.embed_size))
        f2 = self.mlp_f(sent2_linear.view(-1, self.embed_size))

        f1 = f1.view(-1, len1, self.embed_size)
        # batch_size x len1 x hidden_size
        f2 = f2.view(-1, len2, self.embed_size)
        # batch_size x len2 x hidden_size

        score1 = torch.bmm(f1, torch.transpose(f2, 1, 2))
        # e_{ij} batch_size x len1 x len2
        prob1 = F.softmax(score1.view(-1, len2),dim=1).view(-1, len1, len2)
        # batch_size x len1 x len2

        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()
        # e_{ji} batch_size x len2 x len1
        prob2 = F.softmax(score2.view(-1, len1),dim=1).view(-1, len2, len1)
        # batch_size x len2 x len1

        sent1_combine = torch.cat(
            (sent1_linear, torch.bmm(prob1, sent2_linear)), 2)
        # batch_size x len1 x (hidden_size x 2)
        sent2_combine = torch.cat(
            (sent2_linear, torch.bmm(prob2, sent1_linear)), 2)
        # batch_size x len2 x (hidden_size x 2)

        '''sum'''
        g1 = self.mlp_g(sent1_combine.view(-1, 2 * self.embed_size))
        g2 = self.mlp_g(sent2_combine.view(-1, 2 * self.embed_size))
        g1 = g1.view(-1, len1, self.embed_size)
        # batch_size x len1 x hidden_size
        g2 = g2.view(-1, len2, self.embed_size)
        # batch_size x len2 x hidden_size

        sent1_output = torch.sum(g1, 1)  # batch_size x 1 x hidden_size
        sent1_output = torch.squeeze(sent1_output, 1)
        sent2_output = torch.sum(g2, 1)  # batch_size x 1 x hidden_size
        sent2_output = torch.squeeze(sent2_output, 1)

        input_combine = torch.cat((sent1_output, sent2_output), 1)
        # batch_size x (2 * hidden_size)
        h = self.mlp_h(input_combine)
        # batch_size * hidden_size

        h = self.final_linear(h)


        return h

class ClassificationHead(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        logits = self.mlp(hidden_state)
        return logits


def load_model(model, checkpoint, args, verbose=False):
    if checkpoint is None or checkpoint == "None":
        if verbose:
            print('No checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            print('Loading finetuned model from %s' % checkpoint)
        model_state_dict = torch.load(checkpoint)

        model_state_dict = fix_state_dict_namespace(model_state_dict)

        start_model = model
        if (hasattr(model, "transformer")
            and all(not s.startswith('transformer.')
                    for s in model_state_dict.keys())):
            print('Loading transfomer only')
            start_model = model.transformer
        start_model.load_state_dict(model_state_dict)

    return model


def fix_state_dict_namespace(model_state_dict):
    old_keys = []
    new_keys = []
    for t in model_state_dict:
        new_key = t
        if t.startswith('module.'):
            new_key = t.replace('module.', '')
        old_keys.append(t)
        new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)

    return model_state_dict

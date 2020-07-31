import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

def pause():
    programPause = input("Press the <ENTER> key to continue...")


class DVSA(nn.Module):
    'my dvsa rpn interaction'

    def __init__(self, input_feature_dims=2048, input_label_dims=1, embedding_size=512,drop_crob=0.9):

        'build input parameters'
        super(DVSA, self).__init__()
        self.margin = 0.2
        self.Q=torch.FloatTensor([[1]]).cuda()
        self.embedding_size = embedding_size
        self.train_drop_crob = drop_crob
        self.input_label_dims = input_label_dims
        self.input_feature_dims = input_feature_dims

        self.entity_embeddings = nn.Linear(self.input_label_dims, self.embedding_size)  # .cuda()
        nn.init.xavier_uniform_(self.entity_embeddings.weight)

        self.proposal_feat = nn.Linear(self.input_feature_dims, self.embedding_size)  # .cuda()
        nn.init.xavier_uniform_(self.proposal_feat.weight)
        self.dropout_query = nn.Dropout(p=0.1)
        self.dropout_ref = nn.Dropout(p=0.1)

    def _feat_extract(self,R):
        rpn_feats = F.tanh(self.proposal_feat(R))
        rpn_feats = self.dropout_ref(rpn_feats)
        rpn_feats=F.normalize(rpn_feats, p=2, dim=-1)
        return rpn_feats

    def _compute_score(self,Embedding_R,Embedding_Q):
        Embedding_Q=Embedding_Q.permute(1,0)
        score=Embedding_R.mm(Embedding_Q)
        score=torch.max(score,0)[0]
        return score
    def _compute_test_score(self,Embedding_R,Embedding_Q):
        Embedding_Q=Embedding_Q.permute(1,0)
        score=Embedding_R.mm(Embedding_Q)
        return score

    def margin_loss(self, Sp, Sn):
        loss = F.relu(self.margin + Sn - Sp)
        return torch.mean(loss)

    def forward(self, input_pos_feature,  input_neg_feature, ):
        embedding_pr = self._feat_extract(input_pos_feature)
        embedding_nr = self._feat_extract(input_neg_feature)
        query_feats = F.tanh(self.entity_embeddings(self.Q))
        query_feats = self.dropout_query(query_feats)
        query_feats = F.normalize(query_feats, p=2, dim=-1)
        Sp= self._compute_score(embedding_pr,query_feats)
        Sn = self._compute_score(embedding_nr,query_feats)
        total_loss=self.margin_loss(Sp, Sn)
        return total_loss

    def score(self,feature):
        feature=self._feat_extract(feature)
        query_feats = F.tanh(self.entity_embeddings(self.Q))
        query_feats = self.dropout_query(query_feats)
        query_feats = F.normalize(query_feats, p=2, dim=-1)
        S=self._compute_test_score(feature,query_feats)
        return S.permute(1,0)

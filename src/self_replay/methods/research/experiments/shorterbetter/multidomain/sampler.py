"""Equal domain exposure in every generation batch, preserving TRL repetition."""
import torch
from torch.utils.data import Sampler
from .prepare import DOMAINS

class BalancedRepeatSampler(Sampler):
    def __init__(self,data_source,mini_repeat_count,batch_size,repeat_count,seed,cover_all=False):
        self.groups={d:[] for d in DOMAINS}
        for i,d in enumerate(data_source['task']):self.groups[d].append(i)
        if batch_size%4 or not all(self.groups.values()):raise ValueError('Four nonempty domains and divisible batch required')
        self.per=batch_size//4;self.batch_size=batch_size;self.mini=mini_repeat_count;self.repeat=repeat_count
        self.generator=torch.Generator().manual_seed(seed)
        self.cover_all=cover_all
        if cover_all and min(map(len,self.groups.values()))<self.per:
            raise ValueError('Full coverage requires enough unique questions per domain per batch')
        self.batches=((max(map(len,self.groups.values()))+self.per-1)//self.per if cover_all
                      else min(len(g)//self.per for g in self.groups.values()))
    def __iter__(self):
        pools={d:[g[i] for i in torch.randperm(len(g),generator=self.generator).tolist()] for d,g in self.groups.items()}
        if self.cover_all:
            for d,g in self.groups.items():
                while len(pools[d]) < self.batches*self.per:
                    cycle=[g[i] for i in torch.randperm(len(g),generator=self.generator).tolist()]
                    tail=set(pools[d][len(pools[d])//self.per*self.per:])
                    # Move boundary duplicates later in the new cycle, keeping
                    # every global batch's questions unique without losing rows.
                    pools[d].extend([i for i in cycle if i not in tail]+[i for i in cycle if i in tail])
        for b in range(self.batches):
            chunk=[]
            for d in DOMAINS:chunk.extend(pools[d][b*self.per:(b+1)*self.per])
            chunk=[chunk[i] for i in torch.randperm(len(chunk),generator=self.generator).tolist()]
            for _ in range(self.repeat):
                for i in chunk:
                    for _ in range(self.mini):yield i
    def __len__(self):return self.batches*self.batch_size*self.mini*self.repeat

import torch
import numpy as np
from .agent import agent

class social_network():

    def __init__(self, k_agents: int, connections_per_agent: int, vocab_size: int, semantic_dimensions: int, starting_observations: int=10, starting_uncertainty: float=.2):
        super(social_network, self).__init__()
        self.agents = [
            agent(vocab_size,semantic_dimensions) for _ in range(k_agents)
        ]

        self.graph = {
            i: np.random.choice([j for j in range(k_agents) if j != i], size=(connections_per_agent,), replace=False)
            for i in range(len(self.agents))
        }


    def interaction(self, env, lam: float=3.):
        sel = np.random.choice(len(self.agents))

        utt = self.agents[sel].speak(env, lam)
        self.agents[sel].listen(utt, env)

        for agent in self.graph[sel]:
            self.agents[agent].listen(utt, env)

    def intragroup_similarity_heatmap(self):
        heatmap = []
        for i, agenti in enumerate(self.agents):
            v = agenti.vocab
            row = torch.ones(size=(len(self.agents),)) * 100.
            row[i] = 0.
            for j in self.graph[i]:
                row[j] = ((v-self.agents[j].vocab)**2).sum()
            heatmap += [row]

        return torch.cat(heatmap)

    def similarity_heatmap(self):
        heatmap, k = [], len(self.agents)

        for i in range(k):
            row = torch.zeros(size=(len(self.agents),))
            for j in range(k):
                row[j] = ((self.agents[i].vocab - self.agents[j].vocab) ** 2).sum()
            heatmap += [row.unsqueeze(0)]

        return torch.cat(heatmap, dim=0)

    def intragroup_diff(self, i):
        dif = [
            ((self.agents[i].vocab - self.agents[j].vocab)**2).sum()
            for j in self.graph[i]
        ]

        return torch.cat(dif).mean()

    def intergroup_diff(self, i):
        dif = [
            ((self.agents[i].vocab - self.agents[j].vocab) ** 2).sum()
            for j in [j_ for j_ in range(len(self.agents)) if j_ not in self.graph[i]]
        ]

        return torch.cat(dif).mean()

    def all_agents_dif(self, intergroup: bool=False):
        results = []
        if intergroup:
            for i in range(len(self.agents)):
                results += [self.intergroup_diff(i)]
        else:
            for i in range(len(self.agents)):
                results += [self.intragroup_diff(i)]

        return torch.cat(results)

    def shuffle_connections(self, keep_connections: int=0, change_no_connections=None):
        for i in range(len(self.graph)):

            connections = np.random.choice(self.graph[i], size=(keep_connections,), replace=False)

            if change_no_connections:
                new_connections = np.random.choice(
                    [j for j in range(len(self.graph)) if j not in connections],
                    size=(change_no_connections-keep_connections,), replace=False
                )
                connections = np.concat([connections, new_connections])

            else:
                new_connections = np.random.choice(
                    [j for j in range(len(self.graph)) if j not in connections],
                    size=(len(self.graph[i]) - keep_connections,), replace=False
                )
                connections = np.concat([connections, new_connections])

            self.graph[i] = connections
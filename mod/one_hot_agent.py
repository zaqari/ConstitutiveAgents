import torch
import numpy as np

def create_ablated_vocabulary(agent, dropout_rate: float=.2):
    mask = torch.nn.Dropout(p=dropout_rate)
    mask = (mask(torch.ones(size=(agent.vocab.shape[0],))).view(-1) != 0).float()

    agent.vocab = agent.vocab * mask.view(-1,1)
    agent.var[~mask.bool()] = .1
    agent.obs = (agent.obs * mask) + (mask == 0).float()

    return agent

def complex_enforcement(agent, additional_permitted_features_per_word: int=1):

    enforcement = agent.enforcement

    additional_axes = [[i, np.random.choice((enforcement[i] == 0).nonzero().view(-1).numpy(), size=(additional_permitted_features_per_word,), replace=False)] for i in range(len(enforcement))]
    for row_no, ones in additional_axes:
        enforcement[row_no,ones] = 1.

    agent.enforcement = enforcement
    return agent

class agent():

    def __init__(self, words_per_feature: int, semantic_dimensions: int, starting_observations: int=10, starting_uncertainty: float=.2, enforcing: bool=True):
        super(agent, self).__init__()

        self.vocab = [torch.eye(semantic_dimensions)] * words_per_feature
        self.vocab = torch.cat(self.vocab, dim=0)
        self.vocab = torch.randn(size=self.vocab.shape) * self.vocab

        self.var = ((self.vocab == 0).float() * .001) + ((self.vocab != 0).float() * .3)
        self.obs = torch.FloatTensor([starting_observations]*self.vocab.shape[0])

        self.enforcement = torch.ones(size=self.vocab.shape)
        if enforcing:
            self.enforcement = self.enforcement * (self.vocab != 0).float()

        self.unk_p = starting_uncertainty

    def __update(self, lexeme, env):

        # update semantic value
        new_obs_mu_update_by = env / (self.obs[lexeme] + 1)
        old_mu_update_by = (self.obs[lexeme] * self.vocab[lexeme]) / (self.obs[lexeme] * (self.obs[lexeme] + 1))
        mu_update_by = new_obs_mu_update_by - old_mu_update_by

        new_mu = self.vocab[lexeme] + (mu_update_by * self.enforcement[lexeme])

        # update variance
        SSE = (self.obs.unsqueeze(-1) * self.var)[lexeme]
        var_update_by = (((env - new_mu)**2) / (self.obs[lexeme] + 1) )
        var_update_by -= SSE / (self.obs[lexeme] * (self.obs[lexeme]+1))
        var_update_by += (self.obs[lexeme] * (mu_update_by**2)) / (self.obs[lexeme] + 1)
        var_update_by = torch.nan_to_num(var_update_by, nan=0.0, posinf=0.0, neginf=0.0)

        new_var = self.var[lexeme] + (var_update_by * self.enforcement[lexeme])

        self.vocab[lexeme] = new_mu
        self.var[lexeme] = new_var
        self.obs[lexeme] += 1

    def __log_likelihood(self, env):
        constant = 1 / (torch.sqrt(2 * torch.pi * self.var))
        observation = ((env - self.vocab) ** 2) / (2 * self.var)
        return constant * (torch.exp(-observation))

    def access_log_like_bits(self, env):
        constant = 1 / (torch.sqrt(2 * torch.pi * self.var))
        observation = ((env - self.vocab) ** 2) / (2 * self.var)
        return constant, observation

    def speak(self, env, lam: float=1.):
        choices = self.__log_likelihood(env)  #* (env != 0).float()
        choices = torch.softmax(choices.sum(dim=-1) * lam, dim=-1)

        return torch.distributions.Categorical(probs=choices).sample(sample_shape=(1,))


    def listen(self, lexeme, env):
        self.__update(lexeme, env)

    def add_vocab_item(self, feature_map):
        self.vocab = torch.cat([self.vocab, torch.zeros(1,self.vocab.shape[-1])], dim=0)
        self.obs = torch.cat([self.obs, torch.FloatTensor([1])], dim=0)
        self.var = torch.cat([self.var, torch.FloatTensor([[self.unk_p] * self.vocab.shape[-1]])])
        self.enforcement = torch.cat([self.enforcement, torch.zeros(size=(1,self.enforcement.shape[-1]))], dim=0)
        self.enforcement[-1,feature_map] = 1.
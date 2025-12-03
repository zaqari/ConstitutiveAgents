import torch

def tap(x):
    print(x)
    print()
    return x

class agent():

    def __init__(self, vocab_size: int, semantic_dimensions: int, starting_observations: int=10, starting_uncertainty: float=.2):
        super(agent, self).__init__()
        self.vocab = torch.randn(size=(vocab_size, semantic_dimensions))
        self.obs = torch.FloatTensor([starting_observations]*vocab_size)
        self.var = torch.FloatTensor([[starting_uncertainty] * semantic_dimensions]*vocab_size)
        self.unk_p = starting_uncertainty

    def __update(self, lexeme, env):
        env_mask = (env != 0).float()

        # update semantic value
        new_obs_mu_update_by = env / (self.obs[lexeme] + 1)
        old_mu_update_by = (self.obs[lexeme] * self.vocab[lexeme]) / (self.obs[lexeme] * (self.obs[lexeme] + 1))
        mu_update_by = new_obs_mu_update_by - old_mu_update_by

        # new_mu = self.vocab[lexeme] + (mu_update_by * env_mask)
        new_mu = self.vocab[lexeme] + mu_update_by

        # update variance
        SSE = (self.obs.unsqueeze(-1) * self.var)[lexeme]
        var_update_by = (((env - new_mu)**2) / (self.obs[lexeme] + 1) )
        var_update_by -= SSE / (self.obs[lexeme] * (self.obs[lexeme]+1))
        var_update_by += (self.obs[lexeme] * (mu_update_by**2)) / (self.obs[lexeme] + 1)
        var_update_by = torch.nan_to_num(var_update_by, nan=0.0, posinf=0.0, neginf=0.0)

        # new_var = self.var[lexeme] + (var_update_by * env_mask)
        new_var = self.var[lexeme] + var_update_by

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

    def add_vocab_item(self):
        self.vocab = torch.cat([self.vocab, torch.zeros(1,self.vocab.shape[-1])], dim=0)
        self.obs = torch.cat([self.obs, torch.FloatTensor([1])], dim=0)
        self.var = torch.cat([self.var, torch.FloatTensor([[self.unk_p] * self.vocab.shape[-1]])])

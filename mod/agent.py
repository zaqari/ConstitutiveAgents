import torch

def tap(x):
    print(x)
    print()
    return x

def create_one_hot_vocabulary(agent, words_per_feature: int=2):
    new_vocab = [torch.eye(agent.vocab.shape[-1]) for _ in  range(words_per_feature)]
    new_vocab = torch.cat(new_vocab, dim=0)
    new_vocab = torch.randn(size=new_vocab.shape) * new_vocab
    new_var = ((new_vocab == 0).float() * .001) + ((new_vocab != 0).float() * .3)
    new_obs = torch.FloatTensor([agent.obs.max()] * new_vocab.shape[0])

    agent.vocab = new_vocab
    agent.var = new_var
    agent.obs = new_obs

    return agent

def create_ablated_vocabulary(agent, words_per_feature: int=2, dropout_rate: float=.2):
    agent = create_one_hot_vocabulary(agent, words_per_feature)

    mask = torch.nn.Dropout(p=dropout_rate)
    mask = (mask(torch.ones(size=(agent.vocab.shape[0],))).view(-1) != 0).float()

    agent.vocab = agent.vocab * mask.view(-1,1)
    agent.var[~mask.bool()] = .1
    agent.obs = (agent.obs * mask) + (mask == 0).float()

    return agent


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
        return torch.log(constant * (torch.exp(-observation)))

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

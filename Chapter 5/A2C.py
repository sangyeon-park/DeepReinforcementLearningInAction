import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gym
import torch.multiprocessing as mp #A


from PIL import Image
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ActorCritic(nn.Module): #B
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4,25)
        self.l2 = nn.Linear(25,50)
        self.actor_lin1 = nn.Linear(50,2)
        self.l3 = nn.Linear(50,25)
        self.critic_lin1 = nn.Linear(25,1)

    def forward(self,x):
        x = F.normalize(x,dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        # l2를 통과한 결과로 actor와 critic을 모두 사용한다 
        # log_softmax(x) := log(softmax(x)) 와 같지만 따로 사용하는 것보다 안정성이 높음 
        actor = F.log_softmax(self.actor_lin1(y),dim=0) #C
        c = F.relu(self.l3(y.detach()))  # critic은 역전파되지 않음 
        # critic은 -1~1 사이의 값을 출력한다 
        # 이러한 값은 보상이 -1, +1인 cartpole과 잘 맞는다 
        critic = torch.tanh(self.critic_lin1(c)) #D
        return actor, critic #E

##### Listing 5.8
def update_params(worker_opt,values,logprobs,rewards,clc=0.1,gamma=0.95):
    # flip: 성분들을 모두 역순으로 정렬
    # view(-1): 평평하게 만듦, 1차원 배열이 아닌 텐서를 넘겨줄 수도 있기 때문 
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1) #A
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)

    Returns = []
    ret_ = torch.Tensor([0])
    # 역순으로 return을 계산해서 Returns 배열에 저장
    for r in range(rewards.shape[0]): #B
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)
    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns,dim=0)
    # actor의 손실 계산, critic의 loss가 역전파되지 않도록 detach 사용 
    actor_loss = -1*logprobs * (Returns - values.detach()) #C
    # critic의 손실은 작을 수록 보상을 더 잘 예측하게 함 
    critic_loss = torch.pow(values - Returns,2) #D
    # critic의 loss 비율은 적절한 비율로 감소시킴 
    loss = actor_loss.sum() + clc*critic_loss.sum() #E
    loss.backward()
    worker_opt.step()
    return actor_loss, critic_loss, len(rewards)

##### Listing 5.7
def run_episode(worker_env, worker_model):
    state = torch.from_numpy(worker_env.env.state).float() #A
    values, logprobs, rewards = [],[],[] #B state value, log probability, reward
    done = False
    j=0
    while (done == False): #C
        j+=1
        policy, value = worker_model(state) #D
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample() #E  확률에 따라 sampling
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        # 뽑은 action에 따라 다음 step 진행 
        state_, _, done, __, info = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()
        if done: #F
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0
        rewards.append(reward)
    return values, logprobs, rewards

# Listing 5.6
def worker(t, worker_model, counter, params, queue):
    # print("t is:",t, flush=True)
    # sys.stdout = open(str(os.getpid()) + ".out", "w")
    # info('function cworker')
    # print ('hello')
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    # 하나의 모형을 모든 프로세스가 공유 
    worker_opt = optim.Adam(lr=1e-4,params=worker_model.parameters()) #A
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        if i%100==0: print(i)
        worker_opt.zero_grad()
        # episode를 진행하여 데이터를 수집하고 매개변수들을 갱신한다 
        values, logprobs, rewards = run_episode(worker_env,worker_model) #B 
        actor_loss,critic_loss,eplen = update_params(worker_opt,values,logprobs,rewards) #C
        counter.value = counter.value + 1 #D
        queue.append((counter.value, len(rewards)))
        # if i%100==0 : print((queue))

        


if __name__ == '__main__':
    MasterNode = ActorCritic()
    MasterNode.share_memory()  # 프로세스들이 model의 매개변수를 복사하는게 아니라 공유하게 함 
    processes = []
    queue = mp.Manager().list([])
    params = {
        'epochs':1000,
        'n_workers':7,
    }
    # 내장 공유 객체를 전역 공유 카운터로 사용
    counter = mp.Value('i',0)  # i는 정수라는 뜻 
    for i in range(params['n_workers']):
        p = mp.Process(target=worker, args=(i,MasterNode,counter, params, queue))
        p.start()
        processes.append(p)

    print(len(processes))
    for p in processes:
        p.join()

    for p in processes:
        p.terminate()
    # 전역 카운터의 값과 첫 프로세스의 종료코드 출력(문제 없으면 0)
    print(counter.value,processes[1].exitcode)


    



    # visualization 
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    env.reset()
    frames = []
    screen = env.render()
    images = [Image.fromarray(screen)]
    for i in range(100):
        
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        logits,value = MasterNode(state)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        # print(env.step(action.detach().numpy()))
        state2, reward, done, _, info = env.step(action.detach().numpy())
        if done:
            print("Lost")
            env.reset()
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        # env.render()
        screen = env.render()
        images.append(Image.fromarray(screen))
        
    env.close()
    
    image_file = 'cartpole-v1.gif'
    # loop=0: loop forever, duration=1: play each frame for 1ms
    images[0].save(image_file, save_all=True, append_images=images[1:], loop=0, duration=1)
    # display_frames_as_gif(frames)

    queue.sort()
    queue = np.array(queue)
    N = 50
    plot_list = []
    print(queue)
    for i in range(len(queue)//N):
        # print(queue[i*N:(i+1)*N,1])
        # print("sum:",np.sum(queue[i*N:(i+1)*N][1]))
        plot_list.append(np.sum(queue[i*N:(i+1)*N,1])/float(N))
        # print(plot_list[-1])
    
    plt.plot(plot_list)
    
    plt.savefig('A2C_result.png')

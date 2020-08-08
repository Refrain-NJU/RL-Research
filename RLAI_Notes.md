# Chapter3:MDP

Finite Markov Decision Processes

+ 给定t-1时刻的state s 和action a，t时刻产生的state s' 和reward r 的概率为：
  $$
  p(s',r|s,a)\doteq Pr\{S_t=s',R_t=r|S_{t-1}=s,A_{t-1}=a\}
  $$
  容易知道，p关于s'和r求和为1。

  从而可以定义state-transition probabilities:
  $$
  p(s'|s,a)\doteq Pr\{S_t=s'|S_{t-1}=s,A_{t-1}=a\}=\sum_{r\in\mathcal{R}} p(s',r|s,a)
  $$
  和expected rewards for state–action pairs: 
  $$
  r(s,a)\doteq\mathbb{E}[R_t|S_{t-1}=s,A_{t-1}=a]=\sum_{r\in\mathcal{R}} r\times p(r|s,a)=\sum_{r\in\mathcal{R}} r\times\sum_{s'\in\mathcal{S}} p(s',r|s,a)
  $$
  它表示了给定s和a，下一时刻的期望收益。

  以及the expected rewards for state–action–next-state triples:
  $$
  r(s,a,s')\doteq\mathbb{E}[R_t|S_{t-1}=s,A_{t-1}=a,S_t=s']=\sum_{r\in\mathcal{R}} r\times p(r|s,a,s')=\sum_{r\in\mathcal{R}} r\times\frac{p(s',r|s,a)}{p(s'|s,a)}
  $$

+ reward hypothesis:
  + That all of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward).
  + 最大化的是**累积奖赏的概率期望**，而不是当前的奖赏

+ 用t时刻后的奖赏序列的某种函数来定义回报：
  $$
  G_t\doteq R_{t+1}+R_{t+2}+\cdots+R_T
  $$

  + 此处定义为奖赏总和，适用于情节式任务。而一个**情节**表示从初始状态开始到终止状态的过程，比如一盘游戏。强化学习任务常分为情节式任务和持续式任务。（Episodic and Continuing Tasks）

  + 引入折扣率$\gamma$：
    $$
    G_t\doteq R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots=\sum_{k=0}^\infin\gamma^kR_{t+k+1}=R_{t+1}+\gamma G_{t+1}
    $$
    当折扣率小于1的时候，只要奖赏序列有界，上式就能收敛，适用于持续性任务。如果定义情节式任务中的终止状态会转移到自己且奖赏为0，则也可用于情节式任务。

+ 策略是从状态到每个动作的选择概率之间的映射，即$\pi(a|s)$表示在状态s下选择动作a的概率。

  + 把策略$\pi$下状态s的价值函数记为$v_\pi(s)$，表示从状态s开始，Agent执行该策略进行决策所获得的回报的期望值：
    $$
    v_\pi(s)\doteq \mathbb{E}_\pi[G_t|S_t=s]=\mathbb{E}_\pi[\sum_{k=0}^\infin\gamma^kR_{t+k+1}|S_t=s]
    $$
    称为策略$\pi$的状态价值函数（state-value function）

  + 把策略$\pi$下在状态s时采用动作a的价值函数记为$v_\pi(s,a)$，表示从状态s开始，执行动作a之后，Agent执行该策略进行决策所获得的回报的期望值
    $$
    q_\pi(s,a)\doteq \mathbb{E}_\pi[G_t|S_t=s,A_t=a]=\mathbb{E}_\pi[\sum_{k=0}^\infin\gamma^kR_{t+k+1}|S_t=s,A_t=a]
    $$
    称为策略$\pi$的动作价值函数（action-value function）
  
+ Bellman方程：

  + $$
    \begin{align}
    v_\pi(s)&=\mathbb{E}_\pi[G_t|S_t=s]\\
    &=\mathbb{E}_\pi[R_{t+1}+\gamma G_{t+1}|S_t=s]\\
    &=\mathbb{E}_\pi[R_{t+1}|S_t=s]+\mathbb{E}_\pi[\gamma G_{t+1}|S_t=s]\\
    &=\sum_{a}\pi(a|s)r(s,a)+\mathbb{E}_\pi[\gamma G_{t+1}|S_t=s]\quad\text{(由定义将r(s,a)展开得到下式)}\\ 
    &=\sum_{a}\pi(a|s)\sum_rr\sum_{s'}p(s',r|s,a)+\gamma\mathbb{E}_\pi[G_{t+1}|S_t=s]\quad\text{(合并求和项得到下式)}\\
    &=\sum_{a}\pi(a|s)\sum_{s',r}p(s',r|s,a)r+\gamma\mathbb{E}_\pi[G_{t+1}|S_t=s]\\
    \end{align}
    $$

    后面那项是表示遵循策略$\pi$，**后一个状态$S_{t+1}$开始的回报的期望**，所以应该遍历所有可能的$S_{t+1}$：
    $$
    \begin{align}
    \mathbb{E}_\pi[G_{t+1}|S_t=s]&=
    \mathbb{E}_\pi(\gamma \sum_{k=0}^\infin\gamma^kR_{t+k+1}|S_t=s)\\
    &=\sum_{a}\pi(a|s)\sum_r\sum_{s'}p(s',r|s,a)\mathbb{E}_\pi[\sum_{k=0}^\infin\gamma^kR_{t+k+1}|S_{t+1}=s']\\
    &=\sum_{a}\pi(a|s)\sum_{s',r}p(s',r|s,a)v_\pi(s')
    \end{align}
    $$
  所以，
    $$
    \begin{align}
    v_\pi(s)&=\sum_a\pi(a|s)\sum_{s'}\sum_{r}p(s',r|s,a)[r+\gamma \mathbb{E}_\pi[G_{t+1}|S_{t+1}=s']]\\
    &=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]\\
    \end{align}
    $$
    
  + 同理，
    $$
    q_\pi(s,a)=\sum_{s',r}p(s',r|s,a)[r+\gamma\sum_{a'}\pi(a'|s')q_\pi(s',a')]
    $$
  
+ 由备份图容易得出:（空心表示状态值，实心表示动作值）
  
  ![image-20200619214253661](pic\image-20200619214253661.png)
  $$
    \begin{align}
    v_\pi(s)&=\sum_a\pi(a|s) q_\pi(s,a)\\
    q_\pi(s,a)&=\sum_{s',r}p(s',r|s,a)[r+\gamma\sum_{a'}\pi(a'|s')q_\pi(s',a')]\\
    &=\sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]
    \end{align}
  $$
  
+ 价值函数定义了策略$\pi$上的一个偏序关系，由此可以定义最优策略、最优状态值函数、最优动作值函数。

  + $$
    v_*(s)\doteq \max_\pi v_\pi(s)\\
    q_*(s,a)\doteq\max_\pi q_\pi(s,a)
    $$

  + **最优策略**：对于一个动力函数而言，总是存在着一个策略$\pi^*$，使得所有的策略都小于等于这个策略
    $$
    \pi^*(a|s)=\begin{cases} 1, & a\in \arg\max_{a'\in\mathcal{A}}Q^*(s,a') \\0, & \text{其他}\end{cases}
    $$

  + 从而，最优状态值函数可以写为
    $$
    \begin{align}
    v_*(s)&\doteq \max_\pi v_\pi(s)\\
    &=\max_\pi\sum_a\pi(a|s) q_\pi(s,a)\\
    &=\max_a q_{\pi_*}(s,a)\\
    &=\max_a \mathbb{E}_{\pi_*}[G_t|S_t=s,A_t=a]\\
    &=\max_a \mathbb{E}_{\pi_*}[R_{t+1}+\gamma G_{t+1}|S_t=s,A_t=a]\\
    &=\max_a \mathbb{E}[R_{t+1}+\gamma v_*(S_{t+1})|S_t=s,A_t=a]\\
    &=\max_a \sum_{s',r}p(s',r|s,a)[r+\gamma v_*(s')]
    \end{align}
    $$
    第三行是因为最优策略的定义，第六行是因为价值函数的定义，对一个期望再求一次期望结果相等：
    $$
    v_\pi(s)\doteq \mathbb{E}_\pi[G_t|S_t=s]
    $$

  + 最优动作值函数可以写为：
    $$
    \begin{align}
    q_*(s,a)&\doteq\max_\pi q_\pi(s,a)\\
    &=\max_\pi \mathbb{E}_\pi[G_t|S_t=s,A_t=a]\\
    &=\max_\pi \mathbb{E}_{\pi}[R_{t+1}+\gamma G_{t+1}|S_t=s,A_t=a]\\
    &=\mathbb{E}[R_{t+1}+\gamma \max_{a'}q_*(S_{t+1},a')|S_t=s,A_t=a]\\
    &=\sum_{s',r}p(s',r|s,a)[r+\gamma\max_{a'}q_*(s',a')]
    \end{align}
    $$
    结合备份图理解：

    ![image-20200619222024011](pic\image-20200619222024011.png)

  + $v_*$用来评估动作的短期结果（即使用$v_*$来做动作的每一步贪心选择），从长期来看贪心策略也是最优的，因为$v_*$已经包含了未来所有可能的行为所产生的回报影响。

# Chapter4:DP

Dynamic Programming on MDP

+ 在有限MDP问题上进行动态规划，一个重要假设是环境模型是完备的。

### Policy Evaluation (Prediction)

+ 策略评估指的是给定一个策略$\pi$，计算它的状态值函数$v_\pi$。

  + $$
    \begin{align}
    v_\pi(s)&\doteq\mathbb{E}_\pi[G_t|S_t=s]\\
    &=\mathbb{E}_\pi[R_{t+1}+G_{t+1|S_t=s}]\\
    &=\mathbb{E}_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s]\\
    &=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]
    \end{align}
    $$

    期望的下标$\pi$表示期望的计算是以遵循$\pi$为条件的。

    当状态转移概率$p(s',r|s,a)$确定时，给定$\pi$，上式就是一个有$|\mathcal{S}|$个变量与$\mathcal{S}$个等式组成的线性方程组。

  + 可以用迭代法求解
    $$
    \begin{align}
    v_{k+1}(s)
    &=\mathbb{E}_\pi[R_{t+1}+\gamma v_k(S_{t+1})|S_t=s]\\
    &=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_k(s')]
    \end{align}
    $$
    ![image-20200620203906664](pic\image-20200620203906664.png)

### Policy Improvement

+ 计算出一个策略的状态值函数后，我们希望改进策略。也就是对于状态s，现有策略已经给出了动作$\pi(s)$，如果按现有策略继续下去，最后得到的就是$v_\pi(s)$，而我们此时是不知道如果选择一个不属于$\pi(s)$的动作a，换一个新策略，会得到更好或更坏的结果。

+ 思路：在状态s选择动作a后，继续遵循现有策略$\pi$计算
  $$
  \begin{align}
  q_\pi(s,a)
  &\doteq\mathbb{E}_\pi[R_{t+1}+\gamma v_k(S_{t+1})|S_t=s,A_t=a]\\
  &=\sum_{s',r}p(s',r|s,a)[r+\gamma v_k(s')]
  \end{align}
  $$
  判断它是否大于还是小于$v_\pi(s)$。我们期望每次遇到状态s时，选择a的结果都会更好，这时候这个新策略就是更好的。

  给定两个策略$\pi,\pi'$，如果对任意s，有$q_\pi(s,\pi'(s))\geq v_\pi(s)$，那么有$v_{\pi'}(s)\geq v_\pi(s)$。其证明过程：

  ![image-20200620205829647](pic\image-20200620205829647.png)

  第二行是定义，第三行的下标$\pi'$指的是从状态s开始遵循策略$\pi'$行动，第四行是上面给的条件$q_\pi(s,\pi'(s))\geq v_\pi(s)$，第五行和第2、3行一样。

  从而，**我们可以评估一个状态中某个特定动作的改变会产生的后果**，因此，可以贪心来选择新策略$\pi'$：
  $$
  \pi'(s)\doteq \arg\max_a q_\pi(s,a)
  $$
  当对所有s，$v_\pi(s)=v_{\pi'}(s)$时，此时就是$v_*$，$\pi,\pi'$都是最优策略。

  这里讨论的是确定性策略，但对于随机策略（指定了状态s处采取行动a的概率）也是成立的。

### Policy Iteration

+ 策略迭代：策略评估->策略改进->策略评估->策略改进->......直到找到最优策略

  ![image-20200620210930587](pic\image-20200620210930587.png)

### Value Iteration

+ One drawback to policy iteration is that each of its iterations involves policy evaluation, which may itself be a protracted iterative computation requiring multiple sweeps through the state set. 

+ 策略迭代中的策略评估理论上在极限处收敛，但显然我们可以提前结束迭代。值迭代就是，**在对每个状态只进行一次更新后就停止策略评估**，即：
  $$
  \begin{align}
  v_{k+1}(s)
  &=\max_a\mathbb{E}[R_{t+1}+\gamma v_k(S_{t+1})|S_t=s,A_t=a]\\
  &=\max_a\sum_{s',r}p(s',r|s,a)[r+\gamma v_k(s')]
  \end{align}
  $$
  在$v_*$存在的情况下，序列$\{v_k\}$都可以收敛到$v_*$

  另一种值迭代理解方式是，它将贝尔曼最优方程作为更新规则
  $$
  \begin{align}
  v_*(s)
  &=\max_a \mathbb{E}[R_{t+1}+\gamma v_*(S_{t+1})|S_t=s,A_t=a]\\
  &=\max_a \sum_{s',r}p(s',r|s,a)[r+\gamma v_*(s')]
  \end{align}
  $$
  ![image-20200620211902692](pic\image-20200620211902692.png)

  在策略迭代中，先使用求和操作取得$V(s)$，再使用argmax操作取得$\pi(s)$，而值迭代中的max操作将这两个结合起来了。

### Asynchronous Dynamic Programming

+ 异步DP可以使用任意的状态值以任意的顺序更新，比如在值迭代中每一轮只更新一个状态的值，那么只要所有状态都在序列$\{v_k\}$中出现无数次，就能够收敛到最优解。



# Chapter5:MC Methods

Monte Carlo Methods

+ MC方法不需要假设拥有完备的环境知识，MC算法只需要经验，从与环境的交互来采样得到状态、动作、收益的序列。

### MCarlo Prediction

+ An obvious way to estimate it from experience, then, is simply to average the returns observed after visits to that state. As more returns are observed, the average should converge to the expected value. This idea underlies all Monte Carlo methods.

+ 假定在策略$\pi$下有经过状态s的多个情节，要估计策略$\pi$下的$v_\pi(s)$，给定一个情节，每一次s的出现称为s的一次访问，第一次出现称为首次访问。

  + First-visit MC prediction：用s的所有首次访问的回报的平均值来估计$v_\pi(s)$。

    ![image-20200621111837023](pic\image-20200621111837023.png)

  + every-visit MC method：使用所有访问的回报的平均值

### MC Estimation of Action Values

+ 在有模型的情况下，单靠状态值函数就能确定一个策略，在无模型情况下，需要显式地确定每个动作的动作值函数。动作值的MC评估就是要估计$q_\pi(s,a)$，和上面一样，只不过换成了状态-动作二元组（s，a），被首次访问或每次访问。
+ maintaining exploration：将指定的state-action二元组作为起点进行采样，同时保证所有state-action二元组都能有非0概率被选为起点，这样就能保证在采样的情节数趋于无穷时，每一个state-action二元组都会被访问无数次。

### MC Control

+ MC控制问题：如何近似最优的策略。基本思想就是广义策略迭代：策略评估$q_\pi(s,a)$，然后在$q_\pi(s,a)$上贪心选择动作获得改进后的策略。

+ 下面是带maintaining exploration的MC，即探索性出发的MC

  ![image-20200621115447893](pic\image-20200621115447893.png)

### MC Control without Exploring Starts

+ On-Policy：用于生成采样数据序列的策略和用于实际决策的待评估和改进的策略相同

  Off-Policy：用于评估或改进的策略与生成采样数据序列的策略不同

+ $\epsilon-$贪心：以$\epsilon$的概率随机选择一个动作，以$1-\epsilon$的概率选择最优动作。从而，所有的非贪心动作被选上的概率是$\frac{\epsilon}{|\mathcal{A}(s)|}$，贪心动作被选中的概率是$1-\epsilon+\frac{\epsilon}{|\mathcal{A}(s)|}$。

+ 软性策略：对任意s,a，有$\pi(a|s)>0$。$\epsilon-$贪心是一种软策略：所有$\pi(a|s)\geq\frac{\epsilon}{|\mathcal{A}(s)|}$。由于缺乏探索性出发假设， 不能简单地通过对当前值函数进行贪心优化来获得改进后的策略，否则就无法进一步试探非贪心的动作。但是，GPI does not require that the policy be taken all the way to a greedy policy, only that it be moved toward a greedy policy。可以对任意一个$\epsilon-$软策略$\pi$，根据$q_\pi$来生成任意一个$\epsilon-$贪心策略保证优于等于$\pi$就行了。

  ![image-20200621125707624](pic\image-20200621125707624.png)

  假设$\pi'$是一个$\epsilon-$贪心策略，$\pi$是一个$\epsilon-$软策略，下面证明$\pi'\geq\pi$：
  $$
  \begin{align}
  q_\pi(s,\pi'(s))&=\sum_a\pi'(a|s)q_\pi(s,a)\\
  &=\frac{\epsilon}{|\mathcal{A}(s)|}\sum_aq_\pi(s,a)+(1-\epsilon)\max_a q_\pi(s,a)\\
  &\geq \frac{\epsilon}{|\mathcal{A}(s)|}\sum_a q_\pi(s,a)+(1-\epsilon)\sum_a\frac{\pi(a|s)-\frac{\epsilon}{|\mathcal{A}(s)|}}{1-\epsilon}q_\pi(s,a)\\
  &=\frac{\epsilon}{|\mathcal{A}(s)|}\sum_a q_\pi(s,a)-\frac{\epsilon}{|\mathcal{A}(s)|}\sum_aq_\pi(s,a)+\sum_a\pi(a|s)q_\pi(s,a)\\
  &=v_\pi(s)
  \end{align}
  $$
  第三行是因为$\sum_a(\pi(a|s)-\frac{\epsilon}{|\mathcal{A}(s)|})=1-|\mathcal{A}(s)|\frac{\epsilon}{|\mathcal{A}(s)|}=1-\epsilon$，就相当于给每个$q_\pi(s,a)$赋予了一个非负权重，再求和，会小于其中的最大值，即$\sum_i p(i)q_i\leq\sum_i p(i)\max_i q=\max_iq\sum_ip(i)=\max_i q$。这样，我们就不需要the assumption of exploring starts.

### Off-policy Prediction via Importance Sampling

+ 在上面的on-policy中，它并不学习最优策略的动作值，而是学习一个接近最优而且仍能探索的策略的动作值。另一个想法是，干脆采取两个策略，一个用来学习并成为最优策略（target policy），另一个用来指导行动的产生（behavior policy），这样，我们认为学习所用的数据离开了待学习的目标策略（In this case we say that learning is from data “o↵” the target policy, and the overall process is termed off-policy learning）。

+ the assumption of coverage：对任意$\pi(a|s)>0$，要求$b(a|s)>0$，保证目标策略下的每个动作都能在b下发生。

+ 重要性采样：对回报值根据其在目标策略和行动策略中出现的相对概率进行加权：

  + 给定初始状态$S_t$，state-action轨迹$\{A_t,S_{t+1},...,S_T\}$在$\pi$下发生的概率为
    $$
    \pi(A_t|S_t)p(S_{t+1}|S_t,A_t)\pi(A_{t+1}|S_{t+1})\cdots p(S_T|S_{T-1},A_{T-1})\\
    =\prod_{k=t}^{T-1}\pi(A_k|S_k)p(S_{k+1}|S_k,A_k)
    $$
    因此the importancesampling ratio为：
    $$
    \rho_{t:T-1}=\frac{\prod_{k=t}^{T-1}\pi(A_k|S_k)p(S_{k+1}|S_k,A_k)}{\prod_{k=t}^{T-1}b(A_k|S_k)p(S_{k+1}|S_k,A_k)}=\prod_{k=1}^{T-1}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}
    $$
    我们希望估计目标策略下的期望回报，但我们只有行动策略的回报$G_t$，使用上面的比率可以调整：
    $$
    \mathbb{E}[\rho_{t:T-1}G_t|S_t=s]=v_\pi(s)
    $$

  + 定义$\mathcal{T}(s)$为首次访问或每次访问经过的状态$s$的时刻集合，ordinary importance sampling：
    $$
    V(s)\doteq\frac{\sum_{t\in\mathcal{T}(s)}\rho_{t:T(t)-1} G_t}{|\mathcal{T}(s)|}
    $$

  + weighted importance sampling（分母为0时其值为0）
    $$
    V(s)\doteq\frac{\sum_{t\in\mathcal{T}(s)}\rho_{t:T(t)-1} G_t}{\sum_{t\in\mathcal{T}(s)}\rho_{t:T(t)-1}}
    $$

+ Ordinary importance sampling is unbiased whereas weighted importance sampling is biased

### Incremental Implementation

+ Ordinary importance sampling的增量式实现比较简单，
  $$
  V_{n+1}=V_{n}+\frac{1}{n}(\rho_{t:T(t)-1}G_t-V_n)
  $$

+ weighted importance sampling的增量式实现，假设有一个回报序列$\{G_1,\cdots,G_{n-1}\}$和对应的权重$W_i$，且维护了一个前n个回报对应的权值的累加和$C_n=\sum_{k=1}^{n}W_k,C_0=0$，已经有
  $$
  V_n\doteq \frac{\sum_{k=1}^{n-1}W_kG_k}{\sum_{k=1}^{n-1}W_k}
  $$
  现在加入了一个新回报和权重$G_n,W_n$，则
  $$
  \begin{align}
  V_{n+1}&\doteq\frac{\sum_{k=1}^{n}W_kG_k}{\sum_{k=1}^{n}W_k}\\
  &=\frac{\sum_{k=1}^{n-1}W_kG_k+W_nG_n}{C_n}\\
  &=\frac{(C_n-W_n)V_n+W_nG_n}{C_n}\\
  &=V_n+\frac{W_n}{C_n}[G_n-V_n]\\
  C_{n+1}&=C_n+W_{n+1}
  \end{align}
  $$

+ ![image-20200621215655303](pic\image-20200621215655303.png)

  W刚开始是1，然后从结束时刻向前面逐步计算，因为$\rho_{t:T(t)-1}$是从t时刻开始到结束时刻的比率乘积。

### Off-policy MC Control

+ ![image-20200621215933695](pic\image-20200621215933695.png)



# Chapter6:TD Learning

Temporal-Difference Learning

+ MC方法免模型，用样本经验计算，且不bootstrap（自助，自举，即不通过其他价值的估计来更新自己的价值估计）
+ 时序差分学习像MC方法一样可以直接从与环境互动的经验中学习策略，免模型；与DP方法一样使用bootstrap，可以基于已经得到的其他状态的估计值来更新当前状态的值函数。

### TD Prediction

+ MC方法需要一直等到情节结束后才能确定值函数的增量，此时$G_t$才已知：
  $$
  V(S_t)\leftarrow V(S_t)+\alpha(G_t-V(S_t))
  $$
  而TD方法只需要等到下一个时刻即可，使用观察到的收益与下一个状态的估计值：
  $$
  V(S_t)\leftarrow V(S_t)+\alpha(R_{t+1}+\gamma V(S_{t+1})-V(S_t))
  $$
  这叫做单步TD或TD(0)。

  ![image-20200623194923433](pic\image-20200623194923433.png)

+ $$
  \begin{align}
  v_\pi(s)&\doteq\mathbb{E}_\pi[G_t|S_t=s]\\
  &=\mathbb{E}_\pi[R_{t+1}+G_{t+1|S_t=s}]\\
  &=\mathbb{E}_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s]
  \end{align}
  $$

  + MC方法把第一行式子的估计值作为目标：
    + 因为此式中的期望是未知的，MC通过采样的方法得到回报来代替实际上的期望回报
  + DP把第三行式子的估计值作为目标：
    + 因为真实的$v_\pi(S_{t+1})$未知，使用当前的估计值$V(S_{t+1})$代替
  + TD把第三行式子的估计值作为目标：
    + 它采样得到该式的期望值，并且使用当前的估计值V来代替真实值$v_\pi$。
    + 把$\delta_t\doteq R_{t+1}+\gamma V(S_{t+1})-V(S_t)$叫做TD误差。

### Sarsa:On-policy TD Control

+ On-Policy的TD控制问题中，动作值函数的更新采用：
  $$
  Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha(R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t))
  $$
  叫做SARSA

  ![image-20200623202449831](pic\image-20200623202449831.png)



### Q-learning:Off-policy TD Control

+ off-policy中的TD控制问题，叫做Q学习，动作值函数的更新采用：
  $$
  Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha(R_{t+1}+\gamma\max_a Q(S_{t+1},a)-Q(S_t,A_t))
  $$
  可以看出，Q学习采用了对最优动作值函数的直接近似作为学习目标（max操作），而与用于生成决策序列的行动策略无关；而Sarsa中学习目标的计算需要知道下一时刻的动作$A_{t+1}$，与生成决策序列的行动策略相关。在Q学习中，虽然行动策略会决定哪些state-action对会被更新，但只要所有state-action对能被持续更新，学习过程就能收敛

  ![image-20200623203333969](pic\image-20200623203333969.png)

### Expected Sarsa

+ 考虑将Q学习中的TD误差换为：
  $$
  \begin{align}
  Q(S_t,A_t)&\leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma\mathbb{E}[Q(S_{t+1},A_{t+1})|S_{t+1}]-Q(S_t,A_t)]\\
  &\leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma\sum_a\pi(a|S_{t+1})Q(S_{t+1},a)-Q(S_t,A_t)]
  \end{align}
  $$
  Given the next state, this algorithm moves deterministically in the same direction as Sarsa moves in expectation, and accordingly it is called Expected Sarsa, but follows the schema of Q-learning.

  期望Sarsa消除了因为随机选择下一个动作而产生的方差，在计算上比Sarsa复杂，但表现要优于Sarsa。

  在这里期望Sarsa是On-policy的，但也可采用与目标策略$\pi$不同的策略来生成行为。可以看出，当上式中的目标策略$\pi$是贪心策略时，期望Sarsa就退化成了Q学习。

+ <img src="pic\image-20200623205039777.png" alt="image-20200623205039777" style="zoom:50%;" />

  <img src="pic\image-20200623205103651.png" alt="image-20200623205103651" style="zoom:50%;" />

  三种算法的backup图

### Maximization Bias and Double Learning

+ All the control algorithms that we have discussed so far involve maximization in the construction of their target policies. For example, in Q-learning the target policy is the greedy policy given the current action values, which is defined with a max, and in Sarsa the policy is often "-greedy, which also involves a maximization operation. In these algorithms, a maximum over estimated values is used implicitly as an estimate of the maximum value, which can lead to a significant positive bias. To see why, consider a single state s where there are many actions a whose true values, q(s, a), are all zero but whose estimated values, Q(s, a), are uncertain and thus distributed some above and some below zero. The maximum of the true values is zero, but the maximum of the estimates is positive, a positive bias. We call this maximization bias.

+ 产生最大化偏差的一种说法是，确定价值最大的动作和估计它的价值这两个过程采用了相同的样本。如果将这些样本分成两个集合，用它们来学习两个独立的对真实价值q(a)的估计$Q_1(a),Q_2(a)$，那我们就可以用其中一个来确定最大化的动作，另一个来计算这个动作的价值的估计值。比如，$A^*=\arg\max_a Q_1(a)$，$Q_2(A^*)=Q_2(\arg\max_a Q_1(a))$，由于$\mathbb{E}[Q_2(A^*)]=q(A^*)$，因此这个估计是无偏的。交换一下$Q_1,Q_2$的位置，就可以得到另一个无偏估计。这就是double learning的思想。

+ 在行动策略中，可以使用两个估计值的和或者平均值。以下是使用两个估计值的和的双Q学习：

  ![image-20200623210928080](pic\image-20200623210928080.png)



# Chapter7:n-step Bootstrapping

+ n步时序差分方法能结合MC方法和时序差分方法，是两者的折中

### n-step TD Prediction

+ MC方法可看作是T（终止时刻）步TD，原始TD方法可看作是1步TD，n步TD预测就是使用多于1个时刻的奖赏、少于到终止时刻的奖赏：

  ![image-20200624110320905](pic\image-20200624110320905.png)

+ 考虑state-reward序列$S_t,R_{t+1},S_{t+1},\cdots,R_T,S_T$，

  + 在MC方法中，$v_\pi(S_t)$的估计值沿着完整回报的方向更新：
    $$
    G_t\doteq R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots+\gamma^{T-t-1}R_T
    $$

  + 在单步TD中，$v_\pi(S_t)$更新的目标是单步回报：
    $$
    G_{t:t+1}\doteq R_{t+1}+\gamma V_t(S_{t+1})
    $$
    $V_t:\mathcal{S}\rightarrow\mathbb{R}$是在t时刻$v_\pi$的估计值，$\gamma V_t(S_{t+1})$代替了$\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots+\gamma^{T-t-1}R_T$

  + 同理，任意n步更新的目标是n步回报：
    $$
    G_{t:t+n}\doteq R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^n V_{t+n-1}(S_{t+n}),\quad n\geq 1,0\leq t<T-n
    $$
    所有的n步回报都可看作完整回报的近似：在n步后截断得到n步回报，剩下部分用$V_{t+n-1}(S_{t+n})$来代替。当$t+n\geq T$时，剩下部分为0，等于$G_t$。

+ 基于n步回报的状态值函数更新公式为：
  $$
  V_{t+n}(S_t)\doteq V_{t+n-1}(S_t)+\alpha[G_{t:t+n}-V_{t+n-1}(S_t)],\quad 0\leq t<T
  $$
  对于其他任何状态s（$s\neq S_t$）的状态值估计不变。

  ![image-20200624112209150](pic\image-20200624112209150.png)

  在此算法中：

  + 对于一个情节，它在实际的前n-1步是不会执行更新的（即不执行$\tau\geq0$那一部分代码），智能体在时间刻$t=n-1$时，开始进行状态值更新，从时刻0处的状态值开始更新。但它结束更新的时刻并不是实际的终止时刻，而是在终止时刻后再更新一段时间直到$\tau==T-1$，此时t仍然在递增，但不会执行$t<T$那一部分代码，此后的$G$就是$\sum_{i=\tau+1}^{T}\gamma^{i-\tau-1}R_i$，即完整回报。



### n-step Sarsa

+ 将n步方法和Sarsa结合：

  回报定义为：
  $$
  G_{t:t+n}\doteq R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^n Q_{t+n-1}(S_{t+n},A_{t+n}),\quad n\geq 1,0\leq t<T-n
  $$
  当$t+n\geq T$时，$G_{t:t+n}=G_t$，

  更新公式为：
  $$
  Q_{t+n}(S_t,A_t)\doteq Q_{t+n-1}(S_t,A_t)+\alpha[G_{t:t+n}-Q_{t+n-1}(S_t,A_t)],\quad 0\leq t<T
  $$
  对于所有$s\neq S_t,a\neq A_t$的s,a，其Q值不变。

  其伪代码为：

  ![image-20200624114705735](pic\image-20200624114705735.png)

+ 期望Sarsa的n步更新中，
  $$
  G_{t:t+n}\doteq R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^n \overline{V}_{t+n-1}(S_{t+n}),\quad n\geq 1,0\leq t<T-n\\
  \overline{V}_{t+n-1}(S_{t+n})=\sum_a\pi(a|s)Q_t(s,a)
  $$
  当$t+n\geq T$时，$G_{t:t+n}=G_t$。

  ![image-20200624114151273](pic\image-20200624114151273.png)



### n-step Off-policy Learning

+ 考虑策略$\pi,b$的n步的相对概率：
  $$
  V_{t+n}(S_t)\doteq V_{t+n-1}(S_t)+\alpha\rho_{t:t+n-1}[G_{t:t+n}-V_{t+n-1}(S_t)],\quad 0\leq t<T
  $$
  $\rho_{t:t+n-1}$是两种策略采取$A_t,\cdots,A_{t+n}$这n个动作的相对概率
  $$
  \rho_{t:h}\doteq\prod_{k=t}^{\min(h,T-1)}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}
  $$
  动作值的更新公式为：
  $$
  Q_{t+n}(S_t,A_t)\doteq Q_{t+n-1}(S_t,A_t)+\alpha\rho_{t+1:t+n}[G_{t:t+n}-Q_{t+n-1}(S_t,A_t)],\quad 0\leq t<T
  $$
  注意$\rho$的下标，比上面式子加了1，要晚一步，为了得到下一个动作。而在期望Sarsa的版本中应该为$\rho_{t+1:t+n-1}$，因为最后一步不需要考虑具体的动作，而是所有动作的动作值与其概率的加权求和。

  ![image-20200624120510185](pic\image-20200624120510185.png)



### n-step Tree Backup Algorithm

+ n步树回溯算法是一种off-policy算法，且**不需要importance sampling**。

+ ![image-20200624131504741](pic\image-20200624131504741.png)

  具体地，

  + 第一层的所有未被采用的动作叶子结点的权值为$\pi(a|S_{t+1})$，实际执行的动作结点（$A_{t+1}$）无权值，第二层所有未被采用的动作叶子结点的权值为$\pi(A_{t+1}|S_{t+1})\pi(a''|S_{t+2})$，第三层所有未被采用的动作叶子结点的权值为$\pi(A_{t+1}|S_{t+1})\pi(A_{t+2}|S_{t+2})\pi(a'''|S_{t+3})$。

  + 树回溯的单步回报，对于$t<T-1$，有：
    $$
    G_{t:t+1}\doteq R_{t+1}+\gamma\sum_a\pi(a|S_{t+1})Q_t(S_{t+1},a)
    $$
    对于$t<T-2$，树回溯的两步回报为：
    $$
    \begin{align}
    G_{t:t+1}&\doteq R_{t+1}+\gamma\sum_{a\neq A_{t+1}}\pi(a|S_{t+1})Q_t(S_{t+1},a)\\& +\gamma\pi(A_{t+1}|S_{t+1})(R_{t+2}+\gamma\sum_{a}\pi(a|S_{t+2})Q_t(S_{t+2},a))\\
    &=R_{t+1}+\gamma\sum_{a\neq A_{t+1}}\pi(a|S_{t+1})Q_t(S_{t+1},a)+\gamma\pi(A_{t+1}|S_{t+1})G_{t+1:t+2}\\
    \end{align}
    $$
    树回溯的n步回报，对于$t<T-1,n\geq2$，有：
    $$
    G_{t:t+n}\doteq R_{t+1}+\gamma\sum_{a\neq A_{t+1}}\pi(a|S_{t+1})Q_{t+n-1}(S_{t+1},a)+\gamma\pi(A_{t+1}|S_{t+1})G_{t+1:t+n}
    $$
    而$G_{T-1:t+n}\doteq R_T$。

    动作值的更新公式为：
    $$
    Q_{t+n}(S_t,A_t)\doteq Q_{t+n-1}(S_t,A_t)+\alpha[G_{t:t+n}-Q_{t+n-1}(S_t,A_t)]
    $$
    其他所有state-action二元组的价值不变。

    ![image-20200624132827154](pic\image-20200624132827154.png)



### n-step Q($\sigma$)

+ 对比n步Sarsa、n步树回溯、n步期望Sarsa：

  ![image-20200624133135461](pic\image-20200624133135461.png)

+ $Q(\sigma)$算法：

  令$\sigma_t\in[0,1]$表示在时刻t采样的程度，$\sigma=1$表示进行采样操作，$\sigma=0$表示求期望而不进行采样，

  考虑$h=t+n$时树回溯算法的n步回报：
  $$
  \begin{align}
  G_{t:h}&\doteq R_{t+1}+\gamma\sum_{a\neq A_{t+1}}\pi(a|S_{t+1})Q_{h-1}(S_{t+1},a)+\gamma\pi(A_{t+1}|S_{t+1})G_{t+1:h}\\
  &=R_{t+1}+\gamma\overline{V}_{h-1}(S_{t+1})-\gamma\pi(A_{t+1}|S_{t+1})Q_{h-1}(S_{t+1},A_{t+1})+\gamma\pi(A_{t+1}|S_{t+1})G_{t+1:h}\\
  &=R_{t+1}+\gamma\pi(A_{t+1}|S_{t+1})(G_{t+1:h}-Q_{h-1}(S_{t+1},A_{t+1}))+\gamma\overline{V}_{h-1}(S_{t+1})
  \end{align}
  $$
  第二行是因为$\overline{V}_{h-1}(S_{t+1})=\sum_a \pi(a|S_{t+1})Q_{h-1}(S_{t+1},a)$。

  在$Q(\sigma)$中，对$t<h\leq T$，有
  $$
  G_{t:h}\doteq R_{t+1}+\gamma[\sigma_{t+1}\rho_{t+1}+(1-\sigma_{t+1})\pi(A_{t+1}|S_{t+1})][G_{t+1:h}-Q_{h-1}(S_{t+1},A_{t+1})]+\gamma \overline{V}_{h-1}(S_{t+1})
  $$
  中间这项即系数要么是$\rho_{t+1}$要么是$\pi(A_{t+1}|S_{t=1})$。

  当这个递归终止时：

  + $h<T$时，$G_{h:h}\doteq Q_{h-1}(S_h,A_h)$
  + $h=T$时，$G_{T-1:T}\doteq R_{T}$

  伪代码如下：

  ![image-20200624135659098](pic\image-20200624135659098.png)



# Chapter8:Planning and Learning with Tabular Methods

+ distribution models：可以生成对所有可能结果的描述及其对应的概率分布

  sample models：从所有可能行中生成一个确定的结果，而这个结果通过概率分布采样得到

+ The heart of both learning and planning methods is the estimation of value functions by backing-up update operations. The di↵erence is that whereas planning uses simulated experience generated by a model, learning methods use real experience generated by the environment.

  一个Q规划：

  ![image-20200625210122684](pic\image-20200625210122684.png)

### Dyna

+ 对于一个规划智能体，实际经验的作用有：

  + 改进模型，使得模型与现实环境更精确地匹配：model learning
  + 直接改善前面几章的价值函数和策略：direct reinforcement learning
  + <img src="pic\image-20200625205754887.png" alt="image-20200625205754887" style="zoom:67%;" />

+ Dyna的架构：

  <img src="pic\image-20200625210525765.png" alt="image-20200625210525765" style="zoom:50%;" />

  Dyna-Q伪代码：

  ![image-20200625210855648](pic\image-20200625210855648.png)

  步骤d是direct RL，e是模型学习，f是规划，即上面的one-step tabular Q-Planning.

  After each transition $S_t,A_t\rightarrow R_{t+1},S_{t+1}$, the model records in its table entry for $S_t,A_t$ the prediction that $R_{t+1},S_{t+1}$ will deterministically follow. Thus, if the model is queried with a state–action pair that has been experienced before, it simply returns the last-observed next state and next reward as its prediction. During planning, the Q-planning algorithm randomly samples only from state–action pairs that have previously been experienced, so the model is never queried with a pair about which it has no information.

  在Dyna-Q中，学习和规划由完全相同的算法完成的，真实经验用于学习，模拟经验用于规划。

  当模型出错的时候：尝试Dyna-Q+：对每一个state-action进行跟踪，记录它上一次与环境真实交互以来过了多长时间，时间越长，越有理由推测这个二元组相关的环境动态特性会产生变化，即关于它的模型是不正确的。假设模型对单步转移的收益是r，这个转移在$\tau$时刻内没有尝试，在更新时就会采用$r+k\sqrt{\tau}$收益。

  

### Prioritized Sweeping

+ 上面的Dyna中，模拟转移是从先前经历过的staet-action中随机采样得到的，这样的效率比较低。对更新进行优先级排序：A queue is maintained of every state–action pair whose estimated value would change nontrivially if updated , prioritized by the size of the change. When the top pair in the queue is updated, the effect on each of its predecessor pairs is computed. If the effect is greater than some small threshold, then the pair is inserted in the queue with the new priority (if there is a previous entry of the pair in the queue, then insertion results in only the higher priority entry remaining in the queue). In this way the effects of changes are efficiently propagated backward until quiescence.

  ![image-20200625212941328](pic\image-20200625212941328.png)



### Expected vs. Sample Updates

+ Focusing for the moment on one-step updates, they vary primarily along three binary dimensions. The first two dimensions are whether they update state values or action values and whether they estimate the value for the optimal policy or for an arbitrary given policy. These two dimensions give rise to four classes of updates for approximating the four value functions, $q^*, v^*, q_\pi, v_\pi$. The other binary dimension is whether the updates are expected updates, considering all possible events that might happen, or sample updates, considering a single sample of what might happen. These three binary dimensions give rise to eight cases, seven of which correspond to specific algorithms, as shown in the figure followed. (The eighth case does not seem to correspond to any useful update.) Any of these onestep updates can be used in planning methods.

  ![image-20200625214109579](pic\image-20200625214109579.png)



# Chapter 9:On-policy Prediction with Approximation
+ 此后将讨论状态空间任意大的情况，我们不期望学习出最优策略或最优值函数，而是找到一个比较好的近似解。
+ 此章讨论function approximation，从已知的策略$\pi$生成的经验来近似一个价值函数，该价值函数是一个具有权重向量$w\in\mathbb{R}^d$的参数化向量。将每一次对状态价值的更新视作一个训练样本，则函数逼近就是有监督学习去逼近真实的状态价值函数。

### The Prediction Objective

+ 状态数通常比权重数大得多，所以指定一个分布$\mu(s)\geq0,\sum_s \mu(s)=1$来表示对每个状态的误差的权重。

  均方价值误差定义为：
  $$
  \overline{VE}(w)\doteq\sum_s\mu(s)[v_\pi(s)-\hat{v}(s,w)]^2
  $$
  其中$v_\pi(s)$是真实价值函数，$\hat{v}(s,w)$是近似价值函数。

  Often $\mu(s)$ is chosen to be the fraction of time spent in s. Under on-policy training this is called the on-policy distribution; we focus entirely on this case in this chapter. In continuing tasks, the on-policy distribution is the stationary distribution under $\pi$.



### Stochastic-gradient and Semi-gradient Methods

+ 随机梯度下降：在离散时刻t=0,1,2,...上对w进行更新。假设在t时刻，观察到一个新样本：状态$S_t$和它真实价值$v_\pi(S_t)$，并假设状态在样本中的分布$\mu$和上面在$\overline{VE}$的分布一样，则权重更新为：
  $$
  w_{t+1}=w_t-\frac{1}{2}\alpha\nabla[v_\pi(S_t)-\hat{v}(S_t,w_t)]^2=w_t+\alpha[v_\pi(S_t)-\hat{v}(S_t,w_t)]\nabla\hat{v}(S_t,w_t),\quad\alpha>0
  $$
  其中，$\nabla\hat{v}(S_t,w_t)$是个列向量。当$\alpha$以满足标准随机近似条件（$\sum_{n=1}^\infin\alpha_n(\alpha)=\infin,\sum_{n=1}^\infin\alpha_n^2(\alpha)<\infin$）的方式减小时，SGD能收敛到局部最优。

  假设第t个训练样本$S_t,U_t$的目标输出$U_t$不是真实的价值$v_\pi(S_t)$，而是它的一个随机近似，则可以用$U_t$来近似取代它：
  $$
  w_{t+1}=w_t+\alpha[U_t-\hat{v}(S_t,w_t)]\nabla\hat{v}(S_t,w_t)
  $$
  当$U_t$是一个无偏估计（$\mathbb{E}[U_t|S_t=s]=v_\pi(S_t)$）时，同上，能收敛到局部最优。

  比如MC更新的目标$G_t$就是真实价值的无偏估计：

  ![image-20200630221817537](pic\image-20200630221817537.png)

+ 如果使用$v_\pi(S_t)$的bootstrap值作为目标，则没有收敛性保证，但在一些情况下可以可靠地收敛。这种方法叫半梯度方法。如半梯度TD(0)使用$U_t\doteq R_{t+1}+\gamma\hat{v}(S_{t+1},w)$作为目标：

  ![image-20200630222132145](pic\image-20200630222132145.png)



### Linear Methods

+ 当近似函数$\hat{v}(\cdot,w)$是w的线性函数时，对应于每个状态s，存在特征向量$x(s)\doteq(x_1(s),\cdots,x_d(s))^T$,
  $$
  \hat{v}(s,w)\doteq w^Tx(s)=\sum_{i=1}^d w_ix_i(s)
  $$
  The vector x(s) is called a **feature vector** representing state s. Each component $x_i(s)$
  of x(s) is the value of a function $x_i : S\rightarrow R$. We think of a feature as the entirety of one of these functions, and we call its value for a state s a feature of s. For linear methods, features are basis functions because they form a linear basis for the set of approximate functions. Constructing d-dimensional feature vectors to represent states is the same as selecting a set of d **basis functions**基函数。

  在SGD中，The gradient of the approximate value function with respect to w in this case is $\nabla\hat{v}(s,w)=x(s)$.

  Therefore
  $$
  w_{t+1}=w_t+\alpha[U_t-\hat{v}(S_t,w_t)]x(S_t)
  $$

+ 对于半梯度方法TD(0)，近似函数为线性时，其收敛性证明较为复杂，见书9.4节。

  线性函数时，TD(0)的更新为：(简写$x_t=x(S_t)$)
  $$
  w_{t+1}=w_t+\alpha(R_{t+1}+\gamma w_t^Tx_{t+1}-w_t^Tx_t)x_t\\
  =w_t+\alpha(R_{t+1}x_t-x_t(x_t-\gamma x_{t+1})^Tw_t)
  $$
  当系统达到稳定状态，对于任意给定的$w_t$，下一个更新的权重向量的期望可以写作：
  $$
  \mathbb{E}[w_{t+1}|w_t]=w_t+\alpha(b-Aw_t),\text{where }b\doteq\mathbb{E}[R_{t+1}x_t]\in\mathbb{R}^d,A\doteq\mathbb{E}[x_t(x_t-\gamma x_{t+1})^T]\in\mathbb{R}^{d\times d}
  $$
  若它收敛，则必须收敛于$w_{TD}$：
  $$
  b-Aw_{TD}=0,\rightarrow w_{TD}=A^{-1}b
  $$
  这个量称为**TD fixed point**.

+ ![image-20200630230326627](pic\image-20200630230326627.png)

  这个算法中，w的更新方程为：
  $$
  w_{t+n}=w_{t+n-1}+\alpha[G_{t:t+n}-\hat{v}(S_t,w_{t+n-1})]\nabla\hat{v}(S_t,w_{t+n-1}),\quad0\leq t<T\\
  G_{t:t+n}\doteq R_{t+1}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^n\hat{v}(S_{t+n},w_{t+n-1}),\quad0\leq t\leq T-n
  $$

+ 关于特征或者说基函数的构造，见书9.5节



### Least-Squares TD

+ 我们可以不用迭代法求解TD不动点，而是通过估计A和b，直接计算TD不动点。

  LSTD首先估计：
  $$
  \hat{A_t}\doteq \sum_{k=0}^{t-1}x_k(x_k-\gamma x_{k+1})^T+\epsilon I\\
  \hat{b_t}\doteq \sum_{k=0}^{t-1}R_{t+1}x_k
  $$
  I为单位矩阵，加上这个偏置项能保证A的估计值可逆。两者不用除以t，是因为在算不动点的时候t消掉了：
  $$
  w_t=\hat{A}_t^{-1}\hat{b}_t
  $$
  求解$\hat{A}_t^{-1}$使用 Sherman-Morrison 公式：
  $$
  \begin{align}
  \hat{A}_t^{-1}&=(\hat{A}_{t-1}+x_{t-1}(x_{t-1}-\gamma x_t)^T)^{-1}\\
  &=\hat{A}_{t-1}^{-1}-\frac{\hat{A}_{t-1}^{-1}x_{t-1}(x_{t-1}-\gamma x_t)^T\hat{A}_{t-1}^{-1}}{1+(x_{t-1}-\gamma x_t)^T\hat{A}_{t-1}^{-1}x_{t-1}}
  \end{align}
  $$
  ![image-20200630234731625](pic\image-20200630234731625.png)



+ 上面讲的参数化学习方法，另一种方法，Memory-based方法是非参数化方法：

  + 保存训练样本到记忆中，而不更新任何参数

  + 当查询一个状态的价值时，从记忆中找出一组样本，用这些样本来计算查询状态的价值（lazy learning）

    比如用KNN方法或加权平均法返回查询状态的价值。而分配权值的函数可以用核函数。



# Chapter10:On-policy Control with Approximation

### Episodic Semi-gradient Control

+ 将半梯度预测方法延申到动作值上：

  此时训练样本成为了$(S_t,A_t,U_t)$，更新目标$U_t$可以是$q_\pi(S_t,A_t)$的任意近似。梯度下降的一般形式是：
  $$
  w_{t+1}\doteq w_t+\alpha[U_t-\hat{q}(S_t,A_t,w_t)]\nabla\hat{q}(S_t,A_t,w_t)
  $$

+ 一个例子：

  单步Sarsa的更新可表示为：
  $$
  w_{t+1}\doteq w_t+\alpha[R_{t+1}+\gamma\hat{q}(S_{t+1},A_{t+1},w_t)-\hat{q}(S_t,A_t,w_t)]\nabla\hat{q}(S_t,A_t,w_t)
  $$
  该方法称之为episodic semi-gradient one-step Sarsa；对于一个固定的策略，其收敛情况和semi-gradient TD(0)一样

  ![image-20200701102813287](pic\image-20200701102813287.png)

  这里，动作的选取可以是 对于当前状态$S_t$的每个可能的动作a，计算$\hat{q}(S_t,a,w_t)$，然后通过一个比如$\epsilon$-greedy策略来选择动作。

  

### Semi-gradient n-step Sarsa

+ 使用n步回报作为目标：
  $$
  G_{t:t+n}\doteq R_{t+1}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^n\hat{q}(S_{t+n},A_{t+n},w_{t+n-1}),\quad0\leq t< T-n\\
  G_{t:t+n}\doteq G_t,\quad t\geq T-n\\
  w_{t+n}=w_{t+n-1}+\alpha[G_{t:t+n}-\hat{q}(S_t,A_t,w_{t+n-1})]\nabla\hat{q}(S_t,A_t,w_{t+n-1}),\quad0\leq t<T
  $$
  Complete pseudocode is given below：

  ![image-20200701103721916](pic\image-20200701103721916.png)

### Average Reward:A New Problem Setting for Continuing Tasks

+ We now introduce a third classical setting—alongside the episodic and discounted settings— for formulating the goal in Markov decision problems (MDPs).

  **average reward setting** applies to continuing problems. 有观点认为，折扣的设定对于函数逼近来说是有问题的，这里的平均奖赏的设定不考虑任何折扣，对于延迟奖赏的重视程度与即时奖赏相同。

  In the average-reward setting, the quality of a policy $\pi$ is defined as the average rate of reward, or simply average reward, while following that policy, which we denote as $r(\pi)$:
  $$
  \begin{align}
  r(\pi)&\doteq \lim_{h\rightarrow\infin}\frac{1}{h}\sum_{t=1}^h\mathbb{E}[R_t|S_0,A_{0:t-1}\sim\pi]\\
  &=\lim_{t\rightarrow\infin}\mathbb{E}[R_t|S_0,A_{0:t-1}\sim\pi]\\
  &=\sum_{s}\mu_\pi(s)\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)r\\
  \mu_\pi(s)&\doteq\lim_{t\rightarrow\infin}Pr\{S_t=s|A_{0:t-1}\sim\pi\}
  \end{align}
  $$
  The second and third equations hold if the steady-state distribution $\mu_\pi(s)$ exists and .
  is independent of $S_0$.

  returns are defined in terms of di↵erences between rewards and the average reward:
  $$
  G_t\doteq R_{t+1}-r(\pi)+R_{t+2}-r(\pi)+\cdots
  $$
  This is known as the **differential return**, and the corresponding value functions are known as **differential value functions**
  $$
  \begin{align}
  v_\pi(s)&=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r-r(\pi)+v_\pi(s')]\\
  q_\pi(s,a)&=\sum_{s',r}p(s',r|s,a)[r-r(\pi)+\sum_{a'}\pi(a'|s')q_\pi(s',a')]\\
  v_*(s)&=\max_a\sum_{s',r}p(s',r|s,a)[r-\max_\pi r(\pi)+v_*(s')]\\
  q_*(s,a)&=\sum_{s',r}p(s',r|s,a)[r-\max_\pi r(\pi)+\max_{a'}q_*(s',a')]\\
  \end{align}
  $$
  对应的两类TD error：
  $$
  \delta_t\doteq R_{t+1}-\overline{R}_{t+1}+\hat{v}(S_{t+1},w_t)-\hat{v}(S_t,w_t)\\
  \delta_t\doteq R_{t+1}-\overline{R}_{t+1}+\hat{q}(S_{t+1},A_{t+1},w_t)-\hat{q}(S_t,A_t,w_t)
  $$
  这里 $\overline{R_t}$是在t时刻对$r(\pi)$的估计。w的更新公式为：
  $$
  w_{t+1}=w_t+\alpha\delta_t\nabla\hat{q}(S_t,A_t,w_t)
  $$
  

  ![image-20200701110350018](pic\image-20200701110350018.png)

+ The root cause of the difficulties  with the discounted control setting is that with function approximation we have lost the policy improvement theorem (Section 4.2). It is no longer true that if we change the policy to improve the discounted value of one state then we are guaranteed to have improved the overall policy in any useful sense. That guarantee was key to the theory of our reinforcement learning control methods. With function approximation we have lost it!

### Differential Semi-gradient n-step Sarsa

+ 将平均奖赏设定推广到n-step bootstrap，n步回报定义为：
  $$
  G_{t:t+n}\doteq R_{t+1}-\overline{R}_{t+1}+\cdots+R_{t+n}-\overline{R}_{t+n}+\hat{q}(S_{t+n},A_{t+n},w_{t+n-1}),\quad0\leq t< T-n\\
  G_{t:t+n}\doteq G_t,\quad t\geq T-n\\
  $$
  n步TD error变为
  $$
  \delta_t\doteq G_{t:t+n}-\hat{q}(S_t,A_t,w_t)
  $$
  更新公式使用
  $$
  w_{t+1}=w_t+\alpha\delta_t\nabla\hat{q}(S_t,A_t,w_t)
  $$
  差分半梯度n步Sarsa的伪代码如下：

  ![image-20200701111514293](pic\image-20200701111514293.png)



# Chapter12:Eligibility Traces

+ 在第7章里用n步时序差分统一了时序差分学习和蒙特卡洛算法，这一章用更优雅的方式统一两者：资格迹。

### The $\lambda-$return 

+ 复合更新目标：如$\frac{1}{2}G_{t:t+2}+\frac{1}{2}G_{t:t+4}$，

+ $\lambda-$回报：所有可能的n步更新的加权平均，每一个的权重为$\lambda^{n-1},\lambda\in[0,1]$，可以知道近期的回报权值较大，每多走一步，权值按$\lambda$倍递减；并乘上$1-\lambda$以使得权值和为1：
  $$
  \begin{align}
  G_t^\lambda&\doteq(1-\lambda)\sum_{n=1}^\infin\lambda^{n-1}G_{t:t+n}\\
  &=(1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}G_{t:t+n}+\lambda^{T-t-1}G_t
  \end{align}
  $$
  可以看出

  + 当$\lambda=1$时，就是常规的回报$G_t$，此时的更新算法就是MC算法
  + 当$\lambda=0$时，就是单步回报$G_{t:t+1}$，此时的更新算法就是单步时序差分算法

  ![image-20200701145614514](pic\image-20200701145614514.png)

+ **off-line $\lambda-$return algorithm** :

  当一个episode结束后，才开始更新权重向量w。使用$\lambda-$回报作为目标：
  $$
  w_{t+1}=w_{t}+\alpha[G_{t}^\lambda-\hat{v}(S_t,w_{t})]\nabla\hat{v}(S_t,w_{t}),\quad t=0,\cdots,T-1
  $$
  n步时序差分算法是前向的：它对于访问的每个状态，向前（未来的方向）探索所有可能的未来奖赏并结合它们；资格迹是后向的：不断利用了历史的奖赏数据



### TD($\lambda$)

+ TD($\lambda$)在一个episode的每一步都更新权重向量，而不是到结束才更新。

  它有一个资格迹向量$z_t\in\mathbb{R}^d$，和权重向量w同维度，是一个短时记忆，影响权重向量w，初始化为0，在每一步累加价值函数的梯度并以$\gamma\lambda$衰减（$\gamma$为折扣）：
  $$
  z_{-1}=0\\
  z_{t}=\gamma\lambda z_{t-1}+\nabla\hat{v}(S_t,w_t),\quad 0\leq t\leq T
  $$
  每一步更新为：
  $$
  w_{t+1}=w_t+\alpha\delta_tz_t\\
  \delta_t\doteq R_{t+1}+\gamma \hat{v}(S_{t+1},w_t)-\hat{v}(S_t,w_t)
  $$
  其伪代码为：

  ![image-20200701151701215](pic\image-20200701151701215.png)

  TD($\lambda$) is oriented backward in time. At each moment we look at the current TD error and assign it backward to each prior state according to how much that state contributed to the current eligibility trace at that time.资格迹归纳了过去所有事件的影响。

  + 当$\lambda=0$，$z_t=\nabla\hat{v}(S_t,w_t)$，退化为第9章里的单步半梯度时序差分更新。
  + 当$\lambda=1$，$z_t$的系数$\gamma$使得它和MC算法的行为一致。



### n-step Truncated $\lambda$-return Methods

+ we define the truncated  $\lambda$-return for time t, given data only up to some later horizon h as:
  $$
  G_{t:h}^\lambda\doteq(1-\lambda)\sum_{n=1}^{h-t-1}\lambda^{n-1}G_{t:t+n}+\lambda^{h-t-1}G_{t:h}
  $$
  类似于第7章中的n步方法，截断TD($\lambda$)更新推迟n步，将所有的k步回报($1\leq k\leq n$)都考虑进来，称为TTD($\lambda$)

  其更新公式为：
  $$
  w_{t+n}=w_{t+n-1}+\alpha[G_{t:t+n}^\lambda-\hat{v}(S_t,w_{t+n-1})]\nabla\hat{v}(S_t,w_{t+n-1}),\quad0\leq t<T
  $$
  ![image-20200701153355247](pic\image-20200701153355247.png)



### Redoing Updates: Online $\lambda$-return Algorithm

+ 如何选择合适的n呢？

  + 在每一个时间步收到一个新的数据时，回到情节的开始，重新设置horizon h，重做一遍更新，比如：

    ![image-20200701154418303](pic\image-20200701154418303.png)

    $w_0^1$由初始化或者上一episode结束得到的权重而来。最终h=T时的权重$w_T^T$会被用在下一个episode中。

  + Online $\lambda$-return 算法的一般更新公式为：
    $$
    w_{t+1}^h=w_{t}^h+\alpha[G_{t:t+h}^\lambda-\hat{v}(S_t,w_{t}^h)]\nabla\hat{v}(S_t,w_{t}^h),\quad0\leq t<h\leq T
    $$
    显然，这个算法计算十分复杂。

### True Online TD($\lambda$)

+ Online $\lambda$-return 算法计算复杂，如何将这样一个forward-view algorithm转换为backward-view algorithm using eligibility traces？

+ 在线 $\lambda$-return 算法的权重向量可以这样写：

   ![image-20200701155322460](pic\image-20200701155322460.png)

  只有对角线上的是我们真正需要的（它们plays a role in bootstrapping in the n-step returns of the updates），因此我们可以取消上标，$w_t\doteq w_t^t$，找到从前一个值计算下一个值的方法，并假设$\hat{v}(s,w)=w^Tx(s)$，得到True Online  TD($\lambda$)的更新公式：
  $$
  w_{t+1}=w_t+\alpha\delta_tz_t+\alpha(w_t^Tx_t-w_{t-1}^Tx_t)(z_t-x_t)
  $$
  这里搬出原来的定义：$x_t\doteq x(S_t),\quad\delta_t\doteq R_{t+1}+\gamma \hat{v}(S_{t+1},w_t)-\hat{v}(S_t,w_t)$.

  而资格迹向量被定义为：
  $$
  z_t\doteq\gamma\lambda z_{t-1}+(1-\alpha\gamma\lambda z_{t-1}^Tx_t)x_t
  $$
  这个算法能产生和Online $\lambda$-return 算法一样的权重向量，但计算比它简单得多。

  ![image-20200701160446003](pic\image-20200701160446003.png)

  这里的资格迹被称为荷兰迹（dutch trace），而之前讨论的叫做累积迹（accumulating trace），还有一种叫做替换迹（replacing trace），但它只适用于表格型情况或者特征向量是二值化的情况，现在已经过时了，它的定义是：
  $$
  \text{if } x_{i,t}=1,z_{i,t}\doteq1;\quad\text{else } z_{i,t}\doteq\gamma\lambda z_{i,t-1}
  $$
  



### Sarsa($\lambda$)

+ 将资格迹拓展到动作价值函数中：

  首先n步回报为：
  $$
  G_{t:t+n}\doteq R_{t+1}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^n\hat{q}(S_{t+n},A_{t+n},w_{t+n-1}),\quad0\leq t< T-n\\
  G_{t:t+n}\doteq G_t,\quad t\geq T-n\\
  $$
  类似可以得到截断$\lambda$回报。

  Sarsa($\lambda$)的更新规则为：
  $$
  w_{t+1}\doteq w_t+\alpha\delta_t z_t\\
  \delta_t\doteq R_{t+1}+\gamma\hat{q}(S_{t+1},A_{t+1},w_t)-\hat{q}(S_t,A_t,w_t)
  $$
  其资格迹为：
  $$
  z_{-1}\doteq 0\\
  z_{t}\doteq\gamma\lambda z_{t-1}+\nabla\hat{q}(S_t,A_t,w_t),\quad 0\leq t\leq T
  $$
  对于二值特征也可用替换迹。

  ![image-20200701162155012](pic\image-20200701162155012.png)

  类似可得到True online Sarsa($\lambda$)

  ![image-20200701162514189](pic\image-20200701162514189.png)

  

+ 后面书中讨论了off-policy中如何运用资格迹的问题，包括重要性采样、Q($\lambda$)、TB($\lambda$)（就是tree-back的资格迹版本），有点难，没太仔细看，以后要用的时候再看。



# Chapter13:Policy Gradient Methods

+ 这一章讨论直接学习参数化的策略的方法。基于某种性能度量$J(\theta)$，为了最大化性能指标，更新类似于梯度上升$\theta_{t+1}=\theta_t+\alpha\hat{\nabla J(\theta_t)}$  .



### The Policy Gradient Theorem

+ 在策略梯度方法中，策略可以用任意的方式参数化，只要$\pi(a|s,\theta)$对$\theta$可导就行。

+ If the action space is discrete and not too large, then a natural and common kind of
  parameterization is to form parameterized numerical preferences $h(s, a,\theta)$ for each state-action pair。比如，exponential soft-max distribution：
  $$
  \pi(a|s,\theta)\doteq\frac{e^{h(s,a\theta)}}{\sum_b e^{h(s,b,\theta)}}
  $$
  这样的动作偏好值h可以任意参数化，比如神经网络，或者简单的线性函数$h(s,a\theta)=\theta^Tx(s,a)$。

+ 策略梯度定理：

  考虑情节式任务，define the performance measure as the value of the start state of the episode，假设每一个episode都从一个非随机的状态$s_0$出发：
  $$
  J(\theta)\doteq v_{\pi_\theta}(s_0)
  $$
  $v_{\pi_\theta}$是在策略$\pi$下的真实价值函数，策略由参数$\theta$决定。则在episodic情况下，策略梯度定理表示为：
  $$
  \nabla J(\theta)\propto\sum_s\mu(s)\sum_a q_\pi(s,a)\nabla\pi(a|s,\theta)
  $$
  这里的分布$\mu$是策略$\pi$下的on-policy策略分布，即策略$\pi$下每个zhuagnt出现的概率，详见9.2节。该定理的证明见13.2节。



### REINFORCE:Monte Carlo Policy Gradient

+ $$
  \nabla J(\theta)\propto\sum_s\mu(s)\sum_a q_\pi(s,a)\nabla\pi(a|s,\theta)=\mathbb{E}_\pi[\sum_a q_\pi(S_t,a)\nabla\pi(a|S_t,\theta)]
  $$

  因此，随机梯度上升算法可以写为：
  $$
  \theta_{t+1}=\theta_t+\alpha\sum_a\hat{q}(S_t,a,w)\nabla\pi(a|S_t,\theta)
  $$
  这里的$\hat{q}$是学习得到的$q_\pi$的近似。This algorithm, which has been called an **all-actions method** because its update involves all of the actions。

+ REINFORCE算法推导：
  $$
  \begin{align}
  \nabla J(\theta)&=\mathbb{E}_\pi[\sum_a q_\pi(S_t,a)\nabla\pi(a|S_t,\theta)]\\
  &=\mathbb{E}_\pi[\sum_a \pi(a|S_t,\theta)q_\pi(S_t,a)\frac{\nabla\pi(a|S_t,\theta)}{\pi(a|S_t,\theta)}]\\
  &=\mathbb{E}_\pi[q_\pi(S_t,A_t)\frac{\nabla\pi(A_t|S_t,\theta)}{\pi(A_t|S_t,\theta)}]\text{ (用采样}A_t\sim\pi\text{替换掉a)}\\
  &=\mathbb{E}_\pi[G_t\frac{\nabla\pi(A_t|S_t,\theta)}{\pi(A_t|S_t,\theta)}]
  \end{align}
  $$
  最后一步是因为$\mathbb{E}_\pi[G_t|S_t,A_t]=q_\pi(S_t,A_t)$。

  最后得到的那个量可以通过每一步采样得到，它的期望等于真实的梯度。因此，随机梯度上升写为：
  $$
  \theta_{t+1}=\theta_t+\alpha G_t\frac{\nabla\pi(A_t|S_t,\theta_t)}{\pi(A_t|S_t,\theta_t)}=\theta_t+\alpha G_t\nabla\ln\pi(A_t|S_t,\theta_t)
  $$
  这里的$G_t$意味着这是一个MC算法。每一个增量的更新正比于**回报**和选取动作的概率梯度除以这个概率本身。正比于回报，使得参数向最大化回报的方向更新；反比于选择动作的概率，使得频繁被选择的动作不被占优。

  下面给出的更新算法中带有折扣因子$\gamma$：

  ![image-20200701195314935](pic\image-20200701195314935.png)

  

### REINFORCE with Baseline

+ The policy gradient theorem can be generalized to include a comparison of the action value to an arbitrary baseline $b(s)$：
  $$
  \nabla J(\theta)\propto\sum_s\mu(s)\sum_a (q_\pi(s,a)-b(s))\nabla\pi(a|s,\theta)
  $$
  这个基线可以是任意函数甚至随机变量，只要不与动作a有关，因为减掉的那项为0：
  $$
  \sum_a b(s)\nabla\pi(a|s,\theta)=b(s)\nabla\sum_a\pi(a|s,\theta)=b(s)\nabla1=0
  $$
  此时，随机梯度上升变为：
  $$
  \theta_{t+1}=\theta_t+\alpha (G_t-b(S_t))\frac{\nabla\pi(A_t|S_t,\theta_t)}{\pi(A_t|S_t,\theta_t)}
  $$
  一般加入基线对方差有很大影响，它应该根据状态而变化。一个自然的基线是状态值函数$\hat{v}(S_t,w)$。

  ![image-20200701200347659](pic\image-20200701200347659.png)



### Actor–Critic Methods

+ 带基线的REINFORCE不是行动者-评论家方法，因为但状态值函数只被用作基线，而不是评论家。行动者-评论家方法：进一步引入自助（bootstrapping）的思想，以减少方差和加速学习

+ 单步actor-critic方法用单步回报来代替REINFORCE中的整个回报：
  $$
  \begin{align}
  \theta_{t+1}&=\theta_t+\alpha (G_{t:t+1}-\hat{v}(S_t,w))\frac{\nabla\pi(A_t|S_t,\theta_t)}{\pi(A_t|S_t,\theta_t)}\\
  &=\theta_t+\alpha (R_{t+1}+\gamma\hat{v}(S_{t+1},w)-\hat{v}(S_t,w))\frac{\nabla\pi(A_t|S_t,\theta_t)}{\pi(A_t|S_t,\theta_t)}\\
  &=\theta_t+\alpha\delta_t\frac{\nabla\pi(A_t|S_t,\theta)}{\pi(A_t|S_t,\theta)}
  \end{align}
  $$
  伪代码为：

  ![image-20200701201700060](pic\image-20200701201700060.png)

+ 带资格迹的episodic的行动者-评论家算法：

  ![image-20200701201811170](pic\image-20200701201811170.png)

  

### Policy Gradient for Continuing Problems

+ As discussed in Section 10.3, for continuing problems without episode boundaries we need to define performance in terms of the average rate of reward per time step:
  $$
  \begin{align}
  J(\theta)\doteq r(\pi)&\doteq \lim_{h\rightarrow\infin}\frac{1}{h}\sum_{t=1}^h\mathbb{E}[R_t|S_0,A_{0:t-1}\sim\pi]\\
  &=\lim_{t\rightarrow\infin}\mathbb{E}[R_t|S_0,A_{0:t-1}\sim\pi]\\
  &=\sum_{s}\mu_\pi(s)\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)r\\
  \mu_\pi(s)&\doteq\lim_{t\rightarrow\infin}Pr\{S_t=s|A_{0:t-1}\sim\pi\}
  \end{align}
  $$
  $\mu$是策略$\pi$下的稳定状态的分布，并假设它一定存在且独立于$S_0$。如果一直根据策略$\pi$选择动作，则这个分布会保持不变：
  $$
  \sum_{s}\mu_\pi(s)\sum_a\pi(a|s,\theta)p(s'|s,a)=\mu_\pi(s'),\text{ for all }s'\in\mathcal{S}
  $$
  用于持续性问题的actor-critic算法：

  ![image-20200701202626490](pic\image-20200701202626490.png)

  continuing problems的策略梯度定理证明见书13.6节
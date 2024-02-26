# Ask4Help: Learning to Leverage an Expert for Embodied Tasks
[Related Work]

{Embodied AI}
- Recent advances in embodied AI tasks have prompted the development of methods for soliciting assistance.
- existing approaches often rely on imitation and extensive model modifications
- this paper introduces an additional policy for expert assistance without altering the underlying model, thus streamlining the process
{Active Learning and Perception}
- active learning의 핵심 아이디어는 모델이 가장 많이 혼동하는 샘플 또는 불확실한 샘플에 집중하여 추가적인 레이블링을 요청하여 모델의 성능을 향상시키려는 것
- 이런 아이디어를 활용해서 Ask4Help에서는 언제 expert intervention이 가장 useful한지를 학습한다고 함
<img width="731" alt="Pasted Graphic 52" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/8ce5b910-0fc4-4f34-a708-faa22f2446fc">

[Learning to Leverage Expert’s Help]

{Problem Definition}
- goal is to achieve a good trade-off between asking for a minimal amount of help and maximizing the performance of the agent at its task
- without modifying the weights of the underlying E-AI model
{Ask4Help Policy}
- The output of the ASK4HELP policy determines if the agent or the expert policy would act at a particular time step t.
- The ASK4HELP policy is trained using reinforcement learning, specifically DD-PPO
- It receives two negative penalties, a relatively large one for task failure, and a smaller penalty for requesting expert help. The RL loss tries to balance the trade-off between these two penalties, thereby attempting to avoid failure with minimal expert help.
{Adapting to User Preferences at Inference Time}
- to support a wider range of user preferences, a range of reward configurations R1:N representing potential user preferences with different penalties associated with agent failure are sampled
- Then, agent is trained by sampling rollouts with each reward configuration Ri with uniform probability

# Nesterov Momentum
<p>- 모멘텀 알고리즘은 누적된 과거 그래디언트가 지향하고 있는 어떤 방향을 현재 그래디언트에 보정하려는 방식</p>
<p>- 일종의 관성 또는 가속도</p>
<img src="https://github.com/heayounchoi/Paper-Study/assets/118031423/5d92a1e8-3a99-42c8-925f-acc42efb7a3d" width="50%" height="50%">
<br>
<img src="https://github.com/heayounchoi/Paper-Study/assets/118031423/0d5adc87-76c4-4ede-8cf9-ed2a856a5f43" width="25%" height="25%">
<br>
<br>
<p>- E는 learning rate</p>
<p>- u는 모멘트 효과에 대한 가중치</p>
<p>- Vt는 파라미터 공간의 현재 위치 이전까지 누적된 그래디언트 벡터</p>
<p>- g는 파라미터 공간의 현재 위치에서의 그래디언트</p>
<p>- 현재 위치에서의 그래디언트와 속도(uVt)를 더하면 Vt+1이 됨</p>
<p>- 모멘텀은 그래디언트가 큰 흐름의 방향을 지속하도록 도와줌</p>
<img src="https://github.com/heayounchoi/Paper-Study/assets/118031423/da9a1b60-4efc-4d28-8c43-c6980b028aba" width="50%" height="50%">
<p>- 다음은 nesterov momentum</p>
<img src="https://github.com/heayounchoi/Paper-Study/assets/118031423/8a96e0e3-72f6-439a-9b7b-9aca80c2a27f" width="25%" height="25%">
<p>- 현재 위치의 그래디언트를 이용하는 것이 아니고 현재 위치에서 속도만큼 전진한 후의 그래디언트를 이용함</p>
<p>- 이를 가리켜 선험적으로 혹은 모험적으로 먼저 진행한 후 에러를 교정한다고 표현함</p>
<img src="https://tensorflowkorea.files.wordpress.com/2017/03/ec8aa4ed81aceba6b0ec83b7-2017-03-22-ec98a4eca084-11-40-58.png" width="50%" height="50%">
<img src="https://tensorflowkorea.files.wordpress.com/2017/03/ec8aa4ed81aceba6b0ec83b7-2017-03-22-ec98a4eca084-11-40-49.png" width="50%" height="50%">
<img src="https://tensorflowkorea.files.wordpress.com/2017/03/ec8aa4ed81aceba6b0ec83b7-2017-03-22-ec98a4eca084-11-40-40.png" width="50%" height="50%">
<img src="https://s0.wp.com/latex.php?latex=%5Cmu+v_t+%3D+%5Cmu+%28%5Cmu+v_%7Bt-1%7D+-+%5Cepsilon+g%28%5Ctheta_t%29%29+%5C%5C+%C2%A0%5Ctheta_%7Bt%2B1%7D+%3D+%5Ctheta_t+-+%5Cepsilon+g%28%5Ctheta_t%29+%2B+%5Cmu+v_t+%3D+%5Ctheta_t+-+%5Cepsilon+g%28%5Ctheta_t%29+%2B%C2%A0%5Cmu+%28%5Cmu+v_%7Bt-1%7D+-+%5Cepsilon+g%28%5Ctheta_t%29%29&bg=ffffff&fg=444444&s=0&c=20201002&zoom=4.5" width="50%" height="50%">
<p>- 미래 위치의 그래디언트를 찾기는 어려우니, 트릭을 위와 같이 트릭을 사용함</p>
<p>- θt를 옮기기 전과 후의 값은 확실히 다르지만, nesterov momentum의 경로를 그대로 따르고 있으므로 학습을 많이 반복하여 최적점에 수렴하면 전체 파라미터 공간에서 A 지점과 B 지점은 거의 동일하게 됨</p>
<p>- 따라서 현재 그래디언트와 이전의 속도만 가지고 nesterov momentum의 궤적을 따라갈 수 있음</p>

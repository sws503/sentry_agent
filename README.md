# sentry_agent
for graduation

If you want to see demo : https://youtu.be/F1nufWoGm_0

### Map
![Alt text](/DefeatRoaches.png)

### Performance
|   | epsilon = 1.0  | epsilon = 0 |
| :------------ |:---------------:| -----:        |
| policy_loss      | 123 | 456         |
| value_loss      | 654        |   321         |
| score | 10        |    440         |

### TODO
- [X] 학습이 잘되는지 파악할 수 있게 디버그모드 만들기
- [X] training 모드, enjoy 모드 만들기
- [X] 가능한 action을 역장, no_op, move, attack, stop, hold로 한정짓기
- [X] 학습결과 그래프로 출력하기
- [ ] #1 파수기는 전투참여없이 아군의 전투를 유리하게 풀어가도록 역장 사용하기
- [ ] #2 파수기가 아군유닛과 함께 싸우도록 역장 사용하기
- [ ] #3 대규모 전투에서 다양한 유닛(고위기사, 파수기)들이 마법을 써서 전투를 승리하도록 하기


### Requirements  
```
pysc2 >= 2.0
tensorflow >= 1.12
tqdm 
```


### How to run  

- To enjoy it  
```python -m run --map DefeatRoaches --agent a3c_sentry_agent.ZergAgent --continuation True```  

- To train it  
```python -m run --map DefeatRoaches --agent a3c_sentry_agent.ZergAgent --training True --continuation True --parallel 8 ```


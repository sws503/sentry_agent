# sentry_agent
for graduation

### Map
![Alt text](/DefeatRoaches.png)

### Performance
|   | epsilon = 1.0  | epsilon = 0 |
| :------------ |:---------------:| -----:        |
| policy_loss      | 123 | 456         |
| value_loss      | 654        |   321         |
| score | 10        |    440         |

### TODO
[] 학습이 잘되는지 파악할 수 있게 디버그모드 만들기
[] training 모드, enjoy 모드 만들기
[x] 가능한 action을 역장, no_op, move, attack, stop, hold로 한정짓기
[x] 학습결과 그래프로 출력하기

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
```python -m run --map DefeatRoaches --agent a3c_sentry_agent.ZergAgent --train True --continuation True --parallel 8 ```


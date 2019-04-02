# sentry_agent
for graduation

### Map
![Alt text](/DefeatRoaches.png)



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


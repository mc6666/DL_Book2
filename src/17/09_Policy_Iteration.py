# Grid World迷宮之策略循環(Policy Iteration)
# 載入相關套件
import numpy as np
from lib.envs.gridworld import GridworldEnv

# 環境
env = GridworldEnv()

## 策略評估函數
# 策略評估函數
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    # 狀態值函數初始化
    V = np.zeros(env.nS)
    V1 = np.copy(V)
    while True:
        delta = 0
        # 更新每個狀態值的函數
        for s in range(env.nS):
            v = 0
            # 計算每個行動後的狀態值函數
            for a, action_prob in enumerate(policy[s]):
                # 取得所有可能的下一狀態值
                for  prob, next_state, reward, done in env.P[s][a]:
                    # 狀態值函數公式，依照所有可能的下一狀態值函數加總 
                    v += action_prob * prob * (reward + 
                                   discount_factor * V[next_state])
            # 比較更新前後的差值，取最大值
            delta = max(delta, np.abs(v - V[s]))
            V1[s] = v
        V = np.copy(V1)
        # 若最大差值 < 門檻值，則停止評估
        if delta < theta:
            break
    return np.array(V)

## 策略改善函數
def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    # 計算行動值函數
    def one_step_lookahead(state, V):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    
    # 一開始採隨機策略，往上/下/左/右走的機率(π)均等
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        # 策略評估
        V = policy_eval_fn(policy, env, discount_factor)
        
        # 若要改變策略，會設定 policy_stable = False
        policy_stable = True
        
        for s in range(env.nS):
            # 依 P 選擇最佳行動
            chosen_a = np.argmax(policy[s])
            
            # 計算下一步的行動值函數
            action_values = one_step_lookahead(s, V)
            # 選擇最佳行動
            best_a = np.argmax(action_values)
            
            # 貪婪策略：若有新的最佳行動，修改行動策略
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]
        
        # 如果已無較佳行動策略，則回傳策略及狀態值函數
        if policy_stable:
            return policy, V

# 執行策略循環
policy, v = policy_improvement(env)

# 顯示結果
print("策略機率分配:")
print(policy)
print("")

print("4x4 策略機率分配 (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("4x4 狀態值函數:")
print(v.reshape(env.shape))

# 驗證答案是否正確
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)




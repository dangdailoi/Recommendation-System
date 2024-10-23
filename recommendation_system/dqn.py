import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
import os

# Định nghĩa lớp DeepQNetwork
class DeepQNetwork:
    def __init__(self, state_size, action_size, gamma=0.95, learning_rate=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=2000)  # Replay buffer

        # Xây dựng mô hình mạng nơ-ron
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])  # Đảm bảo state có đúng hình dạng
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Thám hiểm
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Khai thác

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, self.state_size])
            next_state = np.reshape(next_state, [1, self.state_size])
            
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        if os.path.exists(name):
            self.model = load_model(name, custom_objects={'mse': MeanSquaredError()})
            # Tạo lại optimizer sau khi load mô hình
            self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            print(f"Mô hình DQN đã được tải từ {name}")
        else:
            print(f"Không tìm thấy mô hình {name}, bắt đầu huấn luyện từ đầu.")

    def save(self, name):
        self.model.save(name)
        print(f"Mô hình DQN đã được lưu tại {name}")